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
from metaworld_algorithms.config.rl import AlgorithmConfig
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
    
    def setup(self):
        # Initialize diffusion schedule
        self.beta_schedule = jnp.linspace(1e-4, 2e-2, self.num_diffusion_steps)
        self.alpha_schedule = 1.0 - self.beta_schedule
        self.alpha_bar_schedule = jnp.cumprod(self.alpha_schedule)
        
        # Noise prediction network
        layers = []
        for _ in range(self.num_layers):
            layers.extend([
                nn.Dense(self.hidden_dim),
                nn.gelu, # TODO: test silu activation
            ])
        layers.append(nn.Dense(self.action_dim))
        self.noise_net = nn.Sequential(layers)
    
    def __call__(self, 
                 observation: Observation, 
                 timestep: jnp.ndarray, 
                 noisy_action: Action) -> Action:
        """Predict noise to denoise the action."""
        # Simple timestep embedding
        timestep_embed = jnp.sin(timestep * jnp.array([1.0, 2.0, 4.0, 8.0]))
        
        # Concatenate inputs
        inputs = jnp.concatenate([observation, timestep_embed, noisy_action], axis=-1)
        
        return self.noise_net(inputs)
    
    def sample(self, observation: Observation, key: PRNGKeyArray) -> Action:
        """Sample action using diffusion denoising process."""
        # Start with pure noise
        key, noise_key = jax.random.split(key)
        action = jax.random.normal(noise_key, (self.action_dim,))
        
        # Denoise iteratively
        for t in reversed(range(self.num_diffusion_steps)):
            key, noise_key = jax.random.split(key)
            
            # Predict noise
            predicted_noise = self(observation, t, action)
            
            # Denoise step (DDPM)
            alpha_t = self.alpha_schedule[t]
            alpha_bar_t = self.alpha_bar_schedule[t]
            beta_t = self.beta_schedule[t]
            
            if t > 0:
                noise = jax.random.normal(noise_key, action.shape)
                sigma_t = jnp.sqrt(beta_t)
            else:
                noise = 0.0
                sigma_t = 0.0
            
            action = (1.0 / jnp.sqrt(alpha_t)) * (
                action - (beta_t / jnp.sqrt(1.0 - alpha_bar_t)) * predicted_noise
            ) + sigma_t * noise
        
        # Apply tanh to bound actions
        return jnp.tanh(action)


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )

    def __call__(self) -> Float[Array, ""]:
        return jnp.exp(self.log_alpha)


@jax.jit
def _sample_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    action = actor.apply_fn(actor.params, observation, action_key, method="sample")
    return action, key


@jax.jit
def _eval_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray
) -> Float[Array, "... action_dim"]:
    # For evaluation, average multiple samples
    keys = jax.random.split(key, 5)
    actions = jax.vmap(
        lambda k: actor.apply_fn(actor.params, observation, k, method="sample")
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


class DIME(OffPolicyAlgorithm[DIMEConfig]):
    actor: TrainState
    critic: CriticTrainState
    temperature: TrainState
    key: PRNGKeyArray
    
    num_tasks: int = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    policy_tau: float = struct.field(pytree_node=False)
    policy_delay: int = struct.field(pytree_node=False)
    entropy_coefficient: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    _n_updates: int = struct.field(pytree_node=False)

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action(self.actor, observation, self.key)
        return self.replace(key=key), action

    @override
    def eval_action(self, observation: Observation) -> Action:
        return _eval_action(self.actor, observation, self.key)

    @override
    def update(self, batch: ReplayBufferSamples) -> tuple[Self, LogDict]:
        """Update the DIME algorithm."""
        # Update critics
        new_critic, critic_logs = self._update_critic(batch)
        
        logs = critic_logs
        
        # Update actor and temperature (with policy delay)
        if (self._n_updates + 1) % self.policy_delay == 0:
            new_actor, new_temperature, policy_logs = self._update_actor_and_temperature(
                new_critic, batch
            )
            logs.update(policy_logs)
        else:
            new_actor = self.actor
            new_temperature = self.temperature
        
        return (
            self.replace(
                actor=new_actor,
                critic=new_critic,
                temperature=new_temperature,
                _n_updates=self._n_updates + 1,
            ),
            logs,
        )

    def _update_critic(self, batch: ReplayBufferSamples) -> tuple[CriticTrainState, LogDict]:
        """Update critic networks."""
        def critic_loss_fn(critic_params):
            # Current Q-values
            q_values = self.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            
            # Target Q-values using target parameters
            key, action_key = jax.random.split(self.key)
            next_actions = self.actor.apply_fn(
                self.actor.params, batch.next_observations, action_key, method="sample"
            )
            
            target_q_values = self.critic.apply_fn(
                self.critic.target_params, batch.next_observations, next_actions
            )
            target_q = jnp.min(target_q_values, axis=0)
            
            # Add entropy bonus
            temperature = self.temperature.apply_fn(self.temperature.params)
            entropy_bonus = temperature * self.entropy_coefficient
            
            # Compute targets
            targets = batch.rewards + self.gamma * (1 - batch.dones) * (
                target_q + entropy_bonus
            )
            targets = jax.lax.stop_gradient(targets)
            
            # Critic loss
            critic_losses = jnp.mean((q_values - targets[None, :]) ** 2, axis=-1)
            total_loss = jnp.sum(critic_losses)
            
            logs = {
                "critic_loss": total_loss,
                "q_mean": jnp.mean(q_values),
                "target_mean": jnp.mean(targets),
            }
            
            return total_loss, logs
        
        (loss, logs), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            self.critic.params
        )
        new_critic = self.critic.apply_gradients(grads=grads)
        
        # Soft update target parameters
        new_target_params = jax.tree_map(
            lambda target, online: self.tau * online + (1 - self.tau) * target,
            new_critic.target_params,
            new_critic.params,
        )
        new_critic = new_critic.replace(target_params=new_target_params)
        
        return new_critic, logs

    def _update_actor_and_temperature(
        self, critic: CriticTrainState, batch: ReplayBufferSamples
    ) -> tuple[TrainState, TrainState, LogDict]:
        """Update actor (diffusion policy) and temperature."""
        def actor_loss_fn(actor_params):
            # Sample actions from diffusion policy
            key, action_key = jax.random.split(self.key)
            actions = self.actor.apply_fn(
                actor_params, batch.observations, action_key, method="sample"
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
                "actor_loss": actor_loss,
                "q_value_mean": jnp.mean(q_value),
            }
        
        def temperature_loss_fn(temperature_params):
            # Temperature loss (entropy regularization)
            temperature = self.temperature.apply_fn(temperature_params)
            
            # Use the target entropy
            temperature_loss = temperature * (self.entropy_coefficient - self.target_entropy)
            
            return jnp.mean(temperature_loss), {"temperature_loss": jnp.mean(temperature_loss)}
        
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
        actor_key, critic_key, temperature_key, algorithm_key = jax.random.split(
            master_key, 4
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
        dummy_timestep = 0
        dummy_action = jnp.array(
            [env_config.action_space.sample() for _ in range(config.num_tasks)]
        )
        
        actor = TrainState.create(
            apply_fn=actor_net,
            params=actor_net.init(actor_key, dummy_obs[0], dummy_timestep, dummy_action[0]),
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
        temperature_net = Temperature(initial_temperature=config.initial_temperature)
        temperature = TrainState.create(
            apply_fn=temperature_net.apply,
            params=temperature_net.init(temperature_key),
            tx=config.temperature_optimizer_config.spawn(),
        )

        target_entropy = -np.prod(env_config.action_space.shape).item()

        return DIME(
            actor=actor,
            critic=critic,
            temperature=temperature,
            key=algorithm_key,
            num_tasks=config.num_tasks,
            tau=config.tau,
            policy_tau=config.policy_tau,
            policy_delay=config.policy_delay,
            entropy_coefficient=config.entropy_coefficient,
            target_entropy=target_entropy,
            _n_updates=0,
        )
