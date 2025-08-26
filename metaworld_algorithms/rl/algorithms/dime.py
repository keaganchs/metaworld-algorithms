"""DIME (Diffusion-Based Maximum Entropy RL) algorithm implementation."""

import dataclasses
from functools import partial
from typing import Self, override

import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax # TODO: check optax use
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray, PyTree


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

# OK
class MultiTaskTemperature(nn.Module):
    num_tasks: int
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda _: jnp.full(
                (self.num_tasks,), jnp.log(self.initial_temperature)
            ),
        )

    def __call__(
        self, task_ids: Float[Array, "... num_tasks"]
    ) -> Float[Array, "... 1"]:
        return jnp.exp(task_ids @ self.log_alpha.reshape(-1, 1))

# OK
class CriticTrainState(TrainState):
    target_params: FrozenDict | None = None


class DiffusionPolicy(nn.Module):
    """Diffusion-based policy network for DIME."""
    
    action_dim: int
    num_tasks: int
    hidden_dim: int = 256
    num_layers: int = 3
    num_diffusion_steps: int = 16

    # num_task_embedding_observations = num_tasks // 5

    @nn.compact
    def __call__(self, observation: Observation, timestep: jnp.ndarray, noisy_action: Action) -> Action:
        """Predict noise to denoise the action."""
        # Get batch dimensions - works for both batched and unbatched inputs
        batch_shape = observation.shape[:-1]
        
        # Convert timestep to proper shape for broadcasting
        timestep_array = jnp.full(batch_shape + (1,), timestep, dtype=jnp.float32)
        # timestep_embed = jnp.sin(timestep_array * jnp.array([1.0, 2.0, 4.0, 8.0]))

        # Concatenate inputs
        x = jnp.concatenate([observation, noisy_action, timestep_array], axis=-1)

        # Forward pass through the network
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        
        # Final output layer
        noise_prediction = nn.Dense(self.action_dim)(x)
        return noise_prediction

# TODO: check implementation
@partial(jax.jit, static_argnames=["apply_fn", "action_dim"])
def diffusion_sample(
    apply_fn, params, observation: Observation, key: PRNGKeyArray, 
    action_dim: int, beta_schedule: jnp.ndarray, alpha_schedule: jnp.ndarray, 
    alpha_bar_schedule: jnp.ndarray
) -> Action:
    """Efficient DDPM sampling with minimal overhead."""
    batch_shape = observation.shape[:-1]
    num_tasks = batch_shape[0] # TODO: might be more efficient to pass as a static arg
    action_shape = batch_shape + (action_dim,)
    num_steps = len(beta_schedule)
    
    # Start with pure noise
    key, noise_key = jax.random.split(key)
    action = jax.random.normal(noise_key, action_shape)
    
    # Pre-compute all values for efficiency
    sqrt_one_minus_alpha_bar = jnp.sqrt(1.0 - alpha_bar_schedule)
    sqrt_alpha_bar = jnp.sqrt(alpha_bar_schedule)
    sqrt_alpha = jnp.sqrt(alpha_schedule)
    
    # Pre-compute posterior variance for all steps
    alpha_bar_prev = jnp.concatenate([jnp.array([1.0]), alpha_bar_schedule[:-1]])
    posterior_var = beta_schedule * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_schedule)
    sqrt_posterior_var = jnp.sqrt(posterior_var)

    # TODO: check if this is actually jittable
    # @jax.jit
    def single_step(i, carry_state):
        action, rng_key = carry_state
        t = (num_steps - 1 - i)  # Reverse time

        # Predict noise
        predicted_noise = apply_fn(params, observation, t, action)
        
        # Denoise
        action = (action - sqrt_one_minus_alpha_bar[t] * predicted_noise) / sqrt_alpha_bar[t]
        
        # Add noise (except last step) - optimized conditional
        rng_key, step_key = jax.random.split(rng_key)
        noise = jax.random.normal(step_key, action_shape)
        
        # Use pre-computed values and efficient conditional
        noise_scale = jnp.where(t > 0, sqrt_posterior_var[t], 0.0)
        action = action + noise_scale * noise
        
        return action, rng_key
    
    # Use fori_loop with minimal state
    final_action, _ = jax.lax.fori_loop(0, num_steps, single_step, (action, key))
    
    return jnp.tanh(final_action)

# TODO: check static args
@partial(jax.jit, static_argnames=("action_dim"))
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

# TODO: check static args
@partial(jax.jit, static_argnames=("action_dim"))
def _eval_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray,
    action_dim: int, beta_schedule: jnp.ndarray, alpha_schedule: jnp.ndarray, alpha_bar_schedule: jnp.ndarray
) -> Float[Array, "... action_dim"]:
    action = diffusion_sample(
        actor.apply_fn, actor.params, observation, key,
        action_dim, beta_schedule, alpha_schedule, alpha_bar_schedule
    )
    return action

# OK
@dataclasses.dataclass(frozen=True)
class DIMEConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    policy_tau: float = 0.005 # TODO
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

    # OK
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

        # Set up RNG keys
        master_key = jax.random.PRNGKey(seed)
        actor_key, critic_key, temperature_key, algorithm_key = (
            jax.random.split(master_key, 4)
        )

        # (Dummy) Values for initialization
        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)]
        )
        dummy_action = jnp.array(
            [env_config.action_space.sample() for _ in range(config.num_tasks)]
        )
        dummy_task_ids = jnp.array(
            [np.ones((config.num_tasks,)) for _ in range(config.num_tasks)]
        )
        dummy_temperature = jnp.array(
            [np.ones((config.num_tasks,)) for _ in range(config.num_tasks)]
        )
        dummy_timestep = 0
        action_dim = int(np.prod(env_config.action_space.shape))

        # Initialize actor (diffusion policy)
        actor_net = DiffusionPolicy(
            action_dim=action_dim,
            num_tasks=config.num_tasks,
            hidden_dim=config.diffusion_hidden_dim,
            num_layers=config.diffusion_num_layers,
            num_diffusion_steps=config.num_diffusion_steps,
        )
        actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(actor_key, dummy_obs, dummy_timestep, dummy_action),
            tx=config.actor_config.network_config.optimizer.spawn(),
        )

        # Initialize critic
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
        temperature_net = MultiTaskTemperature(config.num_tasks, config.initial_temperature)
        temperature = TrainState.create(
            apply_fn=temperature_net.apply,
            params=temperature_net.init(temperature_key, dummy_task_ids),
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
            "actor_num_params": sum(
                x.size for x in jax.tree.leaves(self.actor.params)
            ),
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
    
    # TODO: up next
    @jax.jit
    def _update_critic(self, 
                       batch: ReplayBufferSamples, 
                       temperature_value: Float[Array, " batch 1"],
                       key: PRNGKeyArray) -> tuple[CriticTrainState, LogDict]:
        """Update critic networks with memory-efficient implementation."""
        def critic_loss_fn(critic_params):
            # Current Q-values
            q_values = self.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )

            # Target Q-values
            # Sample next actions
            next_actions = diffusion_sample(
                self.actor.apply_fn, self.actor.params, batch.next_observations, key,
                self.action_dim, self.beta_schedule, self.alpha_schedule, self.alpha_bar_schedule
            )
            # TODO: add dropout layer?
            target_q_values = self.critic.apply_fn(
                self.critic.target_params, batch.next_observations, next_actions
            )
            target_q = jnp.min(target_q_values, axis=0)
            
            # Compute targets with entropy bonus
            entropy_bonus = temperature_value * self.entropy_coefficient
            targets = jax.lax.stop_gradient(
                batch.rewards + self.gamma * (1 - batch.dones) * (target_q + entropy_bonus)
            )
            
            # Critic loss
            # TODO: check equivilence to DIME
            def binary_cross_entropy(prediction, target) -> Float[Array, ""]:
                return -jnp.mean(target * jnp.log(prediction + 1e-15) + (1 - target) * jnp.log(1 - prediction + 1e-15))

            critic_losses = binary_cross_entropy(q_values, targets[None, :])
            total_loss = jnp.mean(critic_losses)
            
            logs = {
                "train/critic_loss": total_loss,
                "train/critic_q_mean": jnp.mean(q_values),
                "train/critic_target_mean": jnp.mean(targets),
                # "train/temperature": self.temperature,
            }
            # "train/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            # "train/critic_params_norm": jnp.linalg.norm(flat_params_crit),
            
            return total_loss, logs
        
        (_, logs), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            self.critic.params
        )
        
        # Apply gradients
        new_critic = self.critic.apply_gradients(grads=grads)
        
        # Soft update target parameters with explicit tree_map to avoid memory accumulation
        new_target_params = jax.tree.map(
            lambda target, online: self.tau * online + (1 - self.tau) * target,
            new_critic.target_params,
            new_critic.params,
        )
        new_critic = new_critic.replace(target_params=new_target_params)
        
        flat_grads, _ = flatten_util.ravel_pytree(grads)

        return self.replace(critic=new_critic, key=key), {
            **logs, 
            "train/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
        }

    # Alpha
    @jax.jit
    def _update_temperature( 
        self,
        log_probs: Float[Array, " batch"],
        task_ids: Float[npt.NDArray, " batch num_tasks"],
    ) -> tuple[Self, LogDict]:
        def temperature_loss(params: FrozenDict) -> Float[Array, ""]:
            log_alpha: jax.Array
            log_alpha = task_ids @ params["params"]["log_alpha"].reshape(-1, 1)  # pyright: ignore [reportAttributeAccessIssue]
            return (-log_alpha * (log_probs + self.target_entropy)).mean()

        temperature_loss_value, temperature_grads = jax.value_and_grad(temperature_loss)(
            self.temperature.params
        )
        temperature = self.temperature.apply_gradients(grads=temperature_grads)

        return self.replace(temperature=temperature), {
            "train/temperature_loss": temperature_loss_value,
            "train/temperature": jnp.exp(temperature.params["params"]["log_alpha"]).sum(),  # pyright: ignore [reportArgumentType]
        }

    @jax.jit
    def _update_actor(
        self, 
        batch: ReplayBufferSamples, 
        temperature_values: Float[Array, " batch"],
        key: PRNGKeyArray,
    ) -> tuple[TrainState, TrainState, LogDict]:
        """Update actor (diffusion policy)."""
        def actor_loss_fn(actor_params):
            # Sample actions from diffusion policy
            actions = diffusion_sample(
                self.actor.apply_fn, actor_params, batch.observations, key,
                self.action_dim, self.beta_schedule, self.alpha_schedule, self.alpha_bar_schedule
            )
            # Q-values
            q_values = self.critic.apply_fn(
                self.critic.params, batch.observations, actions
            )
            min_q_values = jnp.min(q_values, axis=0)

            # Actor loss (maximize Q-value + entropy)
            actor_loss = -jnp.mean(min_q_values + temperature_values * self.entropy_coefficient)
            
            return actor_loss, {
                "train/actor_loss": actor_loss,
                "train/q_value_mean": jnp.mean(min_q_values),
                # "train/actor_params_norm": jnp.linalg.norm(flat_params_act),
            }
        
        # Update actor
        (actor_loss, actor_logs), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(self.actor.params)
        new_actor = self.actor.apply_gradients(grads=actor_grads)
        
        # Update temperature
        (_, temperature_logs) = self._update_temperature(
            log_probs=temperature_values, task_ids=batch.observations[..., -self.num_tasks:]
        )

        # flattened_grads = flatten_util.ravel_pytree(actor_grads)

        logs = {**actor_logs,
                # "train/actor_grad_magnitude": jnp.linalg.norm(flattened_grads),
                **temperature_logs}
        
        return self.replace(actor=new_actor), logs

    @jax.jit
    def _update_inner(self, batch: ReplayBufferSamples) -> tuple[Self, LogDict]:
        """Update the DIME algorithm."""
        # Split keys for proper randomness in each update
        key1, key2, key3, new_key = jax.random.split(self.key, 4)
        
        # Get task IDs
        task_ids = batch.observations[..., -self.num_tasks :]

        # Update temperature
        temperature_vals = self.temperature.apply_fn(self.temperature.params, task_ids)

        # actor_data = critic_data = batch
        # actor_alpha_vals = critic_alpha_vals = temperature_vals
        alpha_val_indices = None

        # Sample actions for critic update
        sampled_actions = diffusion_sample(
            self.actor.apply_fn, self.actor.params, batch.observations, key3,
            self.action_dim, self.beta_schedule, self.alpha_schedule, self.alpha_bar_schedule
        )
        
        _, critic_logs = self._update_critic(batch, temperature_vals, key1)
        
        logs = {}

        # Only log critic metrics if we're at a logging step
        should_log = (self._n_updates + 1) % self.logging_frequency == 0
        if should_log:
            logs.update(critic_logs)
        
        # Update actor and temperature (with policy delay)
        if (self._n_updates + 1) % self.policy_delay == 0:
            _, policy_logs = self._update_actor(batch, temperature_vals, key2)
            if should_log:
                logs.update(policy_logs)
        
        return self.replace(
                # TODO: check if replacing all at once is more efficient
                # actor=new_actor,
                # critic=new_critic,
                # temperature=new_temperature,
                key=new_key,
                _n_updates=self._n_updates + 1,
            ), logs

    @override
    def update(self, batch: ReplayBufferSamples) -> tuple[Self, LogDict]:
        return self._update_inner(batch)
