"""DIME (Diffusion-Based Maximum Entropy RL) algorithm implementation."""

import dataclasses
from functools import partial
from typing import Self, override

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray

from metaworld_algorithms.config.envs import EnvConfig
from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import AlgorithmConfig, OffPolicyTrainingConfig
from metaworld_algorithms.rl.buffers import ReplayBuffer, MultiTaskReplayBuffer
from metaworld_algorithms.rl.networks import (
    ContinuousActionPolicy,
    Ensemble,
    QValueFunction,
)
from metaworld_algorithms.types import (
    Action,
    Intermediates,
    LayerActivationsDict,
    LogDict,
    Observation,
    ReplayBufferSamples,
)

from .base import OffPolicyAlgorithm


class CriticTrainState(TrainState):
    target_params: FrozenDict | None = None


class DiffusionPolicy(nn.Module):
    """Diffusion-based policy network for DIME."""
    
    action_dim: int
    hidden_dim: int = 256
    num_layers: int = 3
    num_diffusion_steps: int = 16
    
    @nn.compact
    def __call__(self, observation: Observation, timestep: jnp.ndarray, noisy_action: Action) -> distrax.Distribution:
        """Predict noise to denoise the action."""
        # Handle both single and batched inputs
        batch_shape = observation.shape[:-1]  # Get batch dimensions
        
        # Simple timestep embedding - broadcast to match batch shape
        timestep_embed = jnp.sin(timestep * jnp.array([1.0, 2.0, 4.0, 8.0]))
        if batch_shape:
            # Expand timestep_embed to match batch dimensions
            timestep_embed = jnp.broadcast_to(
                timestep_embed, batch_shape + timestep_embed.shape
            )
        
        # Ensure noisy_action matches batch shape
        if batch_shape and noisy_action.shape != observation.shape[:-1] + (self.action_dim,):
            # If action is not batched but observation is, broadcast it
            noisy_action = jnp.broadcast_to(
                noisy_action, batch_shape + (self.action_dim,)
            )
        
        # Concatenate inputs
        inputs = jnp.concatenate([observation, timestep_embed, noisy_action], axis=-1)
        
        # Create the noise prediction network with @nn.compact
        x = inputs
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )

    def __call__(self) -> Float[Array, ""]:
        return jnp.exp(self.log_alpha)


@partial(jax.jit, static_argnums=(0, 4))
def diffusion_sample(
    apply_fn, params, observation: Observation, key: PRNGKeyArray, 
    action_dim: int, beta_schedule: jnp.ndarray, alpha_schedule: jnp.ndarray, alpha_bar_schedule: jnp.ndarray
) -> Action:
    """Sample action using diffusion denoising process."""
    # Handle both single and batched inputs
    batch_shape = observation.shape[:-1]  # Get batch dimensions
    action_shape = batch_shape + (action_dim,)
    
    num_diffusion_steps = len(beta_schedule)
    
    # Pre-split all keys at once to avoid repeated key operations in loop
    keys = jax.random.split(key, num_diffusion_steps + 1)
    
    # Start with pure noise
    action = jax.random.normal(keys[0], action_shape)
    
    # Use jax.fori_loop for efficient JIT compilation
    def denoise_step(i, action_state):
        action = action_state
        t = num_diffusion_steps - 1 - i  # Reverse the order
        step_key = keys[i + 1]  # Use pre-split key
        
        # Predict noise using the network
        predicted_noise = apply_fn(params, observation, t, action)
        
        # Denoise step (DDPM) - pre-compute values to avoid repeated indexing
        alpha_t = alpha_schedule[t]
        alpha_bar_t = alpha_bar_schedule[t]
        beta_t = beta_schedule[t]
        
        # Generate noise conditionally
        noise = jax.random.normal(step_key, action.shape)
        sigma_t = jnp.sqrt(beta_t)
        
        # Apply noise only if t > 0 (vectorized conditional)
        noise_mask = (t > 0).astype(jnp.float32)
        noise = noise * noise_mask
        sigma_t = sigma_t * noise_mask
        
        # Denoising update
        action = (1.0 / jnp.sqrt(alpha_t)) * (
            action - (beta_t / jnp.sqrt(1.0 - alpha_bar_t)) * predicted_noise
        ) + sigma_t * noise
        
        return action
    
    action = jax.lax.fori_loop(0, num_diffusion_steps, denoise_step, action)
    
    # Apply tanh to bound actions
    return jnp.tanh(action)


@partial(jax.jit, static_argnums=(3,))
def _sample_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray,
    action_dim: int, beta_schedule: jnp.ndarray, alpha_schedule: jnp.ndarray, alpha_bar_schedule: jnp.ndarray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    action_key, next_key = jax.random.split(key)
    action = diffusion_sample(
        actor.apply_fn, actor.params, observation, action_key,
        action_dim, beta_schedule, alpha_schedule, alpha_bar_schedule
    )
    return action, next_key


@partial(jax.jit, static_argnums=(3,))
def _eval_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray,
    action_dim: int, beta_schedule: jnp.ndarray, alpha_schedule: jnp.ndarray, alpha_bar_schedule: jnp.ndarray
) -> Float[Array, "... action_dim"]:
    # For evaluation, average multiple samples - use fewer samples to reduce computation
    keys = jax.random.split(key, 3)  # Reduced from 5 to 3 samples
    actions = jax.vmap(
        lambda k: diffusion_sample(
            actor.apply_fn, actor.params, observation, k,
            action_dim, beta_schedule, alpha_schedule, alpha_bar_schedule
        )
    )(keys)
    return jnp.mean(actions, axis=0)


@dataclasses.dataclass(frozen=True)
class DIMEConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    policy_tau: float = 0.005
    num_diffusion_steps: int = 16
    diffusion_hidden_dim: int = 256
    diffusion_num_layers: int = 3
    policy_delay: int = 2
    entropy_coefficient: float = 0.1
    logging_frequency: int = 1000  # How often to log metrics (reduced for performance)


class DIME(OffPolicyAlgorithm[DIMEConfig]):
    actor: TrainState
    critic: CriticTrainState
    temperature: TrainState
    key: PRNGKeyArray
    
    num_tasks: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    policy_tau: float = struct.field(pytree_node=False)
    policy_delay: int = struct.field(pytree_node=False)
    entropy_coefficient: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    num_diffusion_steps: int = struct.field(pytree_node=False)
    # Make schedules part of pytree for proper serialization
    beta_schedule: Array
    alpha_schedule: Array
    alpha_bar_schedule: Array
    _n_updates: int = struct.field(pytree_node=False)
    logging_frequency: int = struct.field(pytree_node=False)

    @override
    @staticmethod
    def initialize(
        config: DIMEConfig,
        env_config: EnvConfig,
        seed: int = 1,
    ) -> "DIME":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        actor_key, critic_key, temperature_key, algorithm_key = (
            jax.random.split(master_key, 4)
        )

        action_dim = int(np.prod(env_config.action_space.shape))

        # Initialize diffusion policy (actor)
        actor_net = DiffusionPolicy(
            action_dim=action_dim,
            hidden_dim=config.diffusion_hidden_dim,
            num_layers=config.diffusion_num_layers,
            num_diffusion_steps=config.num_diffusion_steps,
        )
        
        # Create dummy inputs for multi-task environments
        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)]
        )
        dummy_key = jax.random.split(actor_key, 1)[0]
        
        # Initialize the dummy parameters using the noise prediction signature
        dummy_timestep = 0
        dummy_action = jnp.zeros((action_dim,))
        dummy_params = actor_net.init(actor_key, dummy_obs[0], dummy_timestep, dummy_action)
        
        actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=dummy_params,
            tx=config.actor_config.network_config.optimizer.spawn(),
        )

        # Initialize critic
        dummy_action = jnp.array(
            [env_config.action_space.sample() for _ in range(config.num_tasks)]
        )
        critic_cls = partial(QValueFunction, config=config.critic_config)
        critic_net = Ensemble(critic_cls, num=config.num_critics)
        critic_init_params = critic_net.init(critic_key, dummy_obs, dummy_action)
        critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init_params,
            target_params=critic_init_params,
            tx=config.critic_config.network_config.optimizer.spawn(),
        )

        # Initialize temperature
        temperature_net = Temperature(initial_temperature=config.initial_temperature)
        temperature = TrainState.create(
            apply_fn=temperature_net.apply,
            params=temperature_net.init(temperature_key),
            tx=config.temperature_optimizer_config.spawn(),
        )

        target_entropy = -np.prod(env_config.action_space.shape).item()

        # Initialize diffusion schedules
        beta_schedule = jnp.linspace(1e-4, 2e-2, config.num_diffusion_steps)
        alpha_schedule = 1.0 - beta_schedule
        alpha_bar_schedule = jnp.cumprod(alpha_schedule)

        return DIME(
            actor=actor,
            critic=critic,
            temperature=temperature,
            key=algorithm_key,
            num_tasks=config.num_tasks,
            gamma=config.gamma,
            tau=config.tau,
            policy_tau=config.policy_tau,
            policy_delay=config.policy_delay,
            entropy_coefficient=config.entropy_coefficient,
            target_entropy=target_entropy,
            action_dim=action_dim,
            num_diffusion_steps=config.num_diffusion_steps,
            beta_schedule=beta_schedule,
            alpha_schedule=alpha_schedule,
            alpha_bar_schedule=alpha_bar_schedule,
            _n_updates=0,
            logging_frequency=config.logging_frequency,
        )

    @override
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> MultiTaskReplayBuffer:
        return MultiTaskReplayBuffer(
            total_capacity=config.buffer_size,
            num_tasks=self.num_tasks,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "actor_num_params": sum(x.size for x in jax.tree.leaves(self.actor.params)),
            "critic_num_params": sum(
                x.size for x in jax.tree.leaves(self.critic.params)
            ),
            "temperature_num_params": sum(
                x.size for x in jax.tree.leaves(self.temperature.params)
            ),
        }

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(
            self.actor, observation, self.key, self.action_dim, 
            self.beta_schedule, self.alpha_schedule, self.alpha_bar_schedule
        )
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observation: Observation) -> Action:
        return jax.device_get(_eval_action(
            self.actor, observation, self.key, self.action_dim,
            self.beta_schedule, self.alpha_schedule, self.alpha_bar_schedule
        ))

    @jax.jit
    def _update_critic(self, batch: ReplayBufferSamples, key: PRNGKeyArray) -> tuple[CriticTrainState, LogDict]:
        """Update critic networks."""
        def critic_loss_fn(critic_params):
            # Current Q-values
            q_values = self.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            
            # Target Q-values using target parameters - use fewer diffusion steps for target
            next_actions = diffusion_sample(
                self.actor.apply_fn, self.actor.params, batch.next_observations, key,
                self.action_dim, self.beta_schedule, self.alpha_schedule, self.alpha_bar_schedule
            )
            
            target_q_values = self.critic.apply_fn(
                self.critic.target_params, batch.next_observations, next_actions
            )
            target_q = jnp.min(target_q_values, axis=0)
            
            # Add entropy bonus - cache temperature computation
            temperature = self.temperature.apply_fn(self.temperature.params)
            entropy_bonus = temperature * self.entropy_coefficient
            
            # Compute targets
            targets = batch.rewards + self.gamma * (1 - batch.dones) * (
                target_q + entropy_bonus
            )
            targets = jax.lax.stop_gradient(targets)
            
            # Critic loss - use more stable loss computation
            critic_losses = (q_values - targets[None, :]) ** 2
            total_loss = jnp.mean(critic_losses)  # Changed from sum to mean for stability
            
            logs = {
                "train/critic_loss": total_loss,
                "train/critic_q_mean": jnp.mean(q_values),
                "train/critic_target_mean": jnp.mean(targets),
                "train/temperature": temperature,
            }
            
            return total_loss, logs
        
        (loss, logs), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            self.critic.params
        )
        new_critic = self.critic.apply_gradients(grads=grads)
        
        # Soft update target parameters
        new_target_params = jax.tree.map(
            lambda target, online: self.tau * online + (1 - self.tau) * target,
            new_critic.target_params,
            new_critic.params,
        )
        new_critic = new_critic.replace(target_params=new_target_params)
        
        return new_critic, logs

    @jax.jit
    def _update_actor_and_temperature(
        self, critic: CriticTrainState, batch: ReplayBufferSamples, key: PRNGKeyArray
    ) -> tuple[TrainState, TrainState, LogDict]:
        """Update actor (diffusion policy) and temperature."""
        def actor_loss_fn(actor_params):
            # Sample actions from diffusion policy
            actions = diffusion_sample(
                self.actor.apply_fn, actor_params, batch.observations, key,
                self.action_dim, self.beta_schedule, self.alpha_schedule, self.alpha_bar_schedule
            )
            
            # Q-values for these actions
            q_values = critic.apply_fn(
                critic.params, batch.observations, actions
            )
            q_value = jnp.min(q_values, axis=0)
            
            # Temperature
            temperature = self.temperature.apply_fn(self.temperature.params)
            
            # Actor loss (maximize Q-value + entropy)
            actor_loss = -jnp.mean(q_value + temperature * self.entropy_coefficient)
            
            return actor_loss, {
                "train/actor_loss": actor_loss,
                "train/q_value_mean": jnp.mean(q_value),
            }
        
        def temperature_loss_fn(temperature_params):
            # Temperature loss (entropy regularization)
            temperature = self.temperature.apply_fn(temperature_params)
            
            # Use the target entropy
            temperature_loss = temperature * (self.entropy_coefficient - self.target_entropy)
            
            return jnp.mean(temperature_loss), {
                # "train/temperature_loss": jnp.mean(temperature_loss),
                "train/temperature": temperature,
            }
        
        # Update actor
        (actor_loss, actor_logs), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(self.actor.params)
        new_actor = self.actor.apply_gradients(grads=actor_grads)
        
        # Update temperature
        (temp_loss, temp_logs), temp_grads = jax.value_and_grad(
            temperature_loss_fn, has_aux=True
        )(self.temperature.params)
        new_temperature = self.temperature.apply_gradients(grads=temp_grads)
        
        logs = {**actor_logs, **temp_logs}
        return new_actor, new_temperature, logs

    def _update_inner(self, batch: ReplayBufferSamples) -> tuple[Self, LogDict]:
        """Update the DIME algorithm."""
        # Split keys for proper randomness in each update
        key1, key2, new_key = jax.random.split(self.key, 3)
        
        # Update critics
        new_critic, critic_logs = self._update_critic(batch, key1)
        
        logs = {}
        
        # Only log critic metrics if we're at a logging step
        should_log = (self._n_updates + 1) % self.logging_frequency == 0
        if should_log:
            logs.update(critic_logs)
        
        # Update actor and temperature (with policy delay)
        if (self._n_updates + 1) % self.policy_delay == 0:
            new_actor, new_temperature, policy_logs = self._update_actor_and_temperature(
                new_critic, batch, key2
            )
            if should_log:
                logs.update(policy_logs)
        else:
            new_actor = self.actor
            new_temperature = self.temperature
        
        return (
            self.replace(
                actor=new_actor,
                critic=new_critic,
                temperature=new_temperature,
                key=new_key,  # Update the key
                _n_updates=self._n_updates + 1,
            ),
            logs,
        )

    @override
    def update(self, batch: ReplayBufferSamples) -> tuple[Self, LogDict]:
        return self._update_inner(batch)
