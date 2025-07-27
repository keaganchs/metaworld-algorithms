"""DIME: Diffusion-Based Maximum Entropy Reinforcement Learning"""

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
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray

from metaworld_algorithms.config.envs import EnvConfig
from metaworld_algorithms.config.networks import QValueFunctionConfig
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import AlgorithmConfig, OffPolicyTrainingConfig
from metaworld_algorithms.rl.buffers import ReplayBuffer
from metaworld_algorithms.rl.networks import (
    Ensemble,
    QValueFunction,
    DiffusionMLP
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


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda _: jnp.full((1,), jnp.log(self.initial_temperature)),
        )

    def __call__(self) -> Float[Array, " 1"]:
        return jnp.exp(self.log_alpha)


class CriticTrainState(TrainState):
    target_params: FrozenDict | None = None


def ddpm_schedule(t: Float[Array, "batch"], T: int = 1000) -> tuple[Float[Array, "batch"], Float[Array, "batch"]]:
    """DDPM noise schedule"""
    beta_start = 1e-4
    beta_end = 2e-2
    betas = jnp.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas)
    
    # Get values for timestep t
    alpha_t = alphas_cumprod[t.astype(jnp.int32)]
    alpha_t_prev = jnp.where(t > 0, alphas_cumprod[t.astype(jnp.int32) - 1], 1.0)
    
    return alpha_t, alpha_t_prev


@jax.jit
def _sample_action_ddpm(
    diffusion_net: TrainState,
    observation: Observation,
    key: PRNGKeyArray,
    num_diffusion_steps: int = 5,
    temperature: float = 1.0,
    clip_sample: bool = True,
) -> tuple[Float[Array, "... action_dim"], PRNGKeyArray]:
    """Sample action using DDPM sampling process"""
    batch_size = observation.shape[0]
    # Hack to get action dimension from the diffusion network
    action_dim = diffusion_net.params["params"]["Dense_2"]["bias"].shape[0]  # Get action dim from final layer
    
    # Start from noise
    key, noise_key = jax.random.split(key)
    x = jax.random.normal(noise_key, (batch_size, action_dim)) * temperature
    
    # Reverse diffusion process
    for i in range(num_diffusion_steps - 1, -1, -1):
        t = jnp.full((batch_size,), i / num_diffusion_steps)
        
        # Predict noise
        predicted_noise = diffusion_net.apply_fn(diffusion_net.params, x, t, observation)
        
        # DDPM update step
        alpha_t, alpha_t_prev = ddpm_schedule(t * 1000, 1000)
        alpha_t = alpha_t[..., None]
        alpha_t_prev = alpha_t_prev[..., None]
        
        # Compute mean
        pred_x0 = (x - jnp.sqrt(1 - alpha_t) * predicted_noise) / jnp.sqrt(alpha_t)
        if clip_sample:
            pred_x0 = jnp.clip(pred_x0, -1.0, 1.0)
        
        # Compute posterior mean
        mean = (alpha_t_prev ** 0.5 * pred_x0 + jnp.sqrt(1 - alpha_t_prev) * predicted_noise) / (alpha_t ** 0.5 + alpha_t_prev ** 0.5)
        
        if i > 0:
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, x.shape)
            variance = (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            x = mean + jnp.sqrt(variance) * noise
        else:
            x = mean
    
    if clip_sample:
        x = jnp.clip(x, -1.0, 1.0)
    
    return x, key


@jax.jit
def _eval_action(
    diffusion_net: TrainState, 
    observation: Observation,
    num_diffusion_steps: int = 5,
) -> Float[Array, "... action_dim"]:
    """Deterministic action for evaluation"""
    key = jax.random.PRNGKey(0)  # Fixed seed for deterministic evaluation
    action, _ = _sample_action_ddpm(diffusion_net, observation, key, num_diffusion_steps, temperature=0.0)
    return action


@dataclasses.dataclass(frozen=True)
class DIMEConfig(AlgorithmConfig):
    diffusion_config: VanillaNetworkConfig = VanillaNetworkConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    num_diffusion_steps: int = 5
    diffusion_schedule_steps: int = 1000
    ema_rate: float = 0.995


class DIME(OffPolicyAlgorithm[DIMEConfig]):
    diffusion_net: TrainState
    critic: CriticTrainState
    alpha: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    num_critics: int = struct.field(pytree_node=False)
    num_diffusion_steps: int = struct.field(pytree_node=False)
    diffusion_schedule_steps: int = struct.field(pytree_node=False)

    @override
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> ReplayBuffer:
        return ReplayBuffer(
            capacity=config.buffer_size,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @override
    @staticmethod
    def initialize(config: DIMEConfig, env_config: EnvConfig, seed: int = 1) -> "DIME":
        assert isinstance(env_config.action_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )
        assert isinstance(env_config.observation_space, gym.spaces.Box), (
            "Non-box spaces currently not supported."
        )

        master_key = jax.random.PRNGKey(seed)
        algorithm_key, diffusion_init_key, critic_init_key, alpha_init_key = (
            jax.random.split(master_key, 4)
        )

        action_dim = int(np.prod(env_config.action_space.shape))
        obs_dim = int(np.prod(env_config.observation_space.shape))
        
        # Initialize diffusion model
        diffusion_model = DiffusionMLP(
            action_dim=action_dim,
            config=config.diffusion_config,
        )
        dummy_obs = jnp.array(
            [env_config.observation_space.sample() for _ in range(config.num_tasks)]
        )
        dummy_action = jnp.array(
            [env_config.action_space.sample() for _ in range(config.num_tasks)]
        )
        dummy_time = jnp.ones((config.num_tasks,))
        
        diffusion_net = TrainState.create(
            apply_fn=diffusion_model.apply,
            params=diffusion_model.init(diffusion_init_key, dummy_action, dummy_time, dummy_obs),
            tx=config.diffusion_config.optimizer.spawn(),
        )

        # Initialize critic
        critic_cls = partial(QValueFunction, config=config.critic_config)
        critic_net = Ensemble(critic_cls, num=config.num_critics)
        critic_init_params = critic_net.init(critic_init_key, dummy_obs, dummy_action)
        critic = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_init_params,
            target_params=critic_init_params,
            tx=config.critic_config.network_config.optimizer.spawn(),
        )

        # Initialize temperature
        alpha_net = Temperature(config.initial_temperature)
        alpha = TrainState.create(
            apply_fn=alpha_net.apply,
            params=alpha_net.init(alpha_init_key),
            tx=config.temperature_optimizer_config.spawn(),
        )

        target_entropy = -np.prod(env_config.action_space.shape).item()

        return DIME(
            num_tasks=config.num_tasks,
            diffusion_net=diffusion_net,
            critic=critic,
            alpha=alpha,
            key=algorithm_key,
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=target_entropy,
            num_critics=config.num_critics,
            num_diffusion_steps=config.num_diffusion_steps,
            diffusion_schedule_steps=config.diffusion_schedule_steps,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "diffusion_num_params": sum(x.size for x in jax.tree.leaves(self.diffusion_net.params)),
            "critic_num_params": sum(
                x.size for x in jax.tree.leaves(self.critic.params)
            ),
        }

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, key = _sample_action_ddpm(
            self.diffusion_net, observation, self.key, self.num_diffusion_steps
        )
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observations: Observation) -> Action:
        return jax.device_get(_eval_action(self.diffusion_net, observations, self.num_diffusion_steps))

    @jax.jit
    def _update_inner(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        key, diffusion_loss_key, critic_loss_key = jax.random.split(self.key, 3)
        
        # Sample next actions using current diffusion model for critic target
        next_actions, _ = _sample_action_ddpm(
            self.diffusion_net, data.next_observations, critic_loss_key, self.num_diffusion_steps
        )

        def update_critic(
            _critic: CriticTrainState,
            alpha_val: Float[Array, "batch 1"],
        ) -> tuple[CriticTrainState, LogDict]:
            # Compute target Q values
            q_values = self.critic.apply_fn(
                self.critic.target_params, data.next_observations, next_actions
            )

            def critic_loss(params: FrozenDict) -> tuple[Float[Array, ""], Float[Array, ""]]:
                min_qf_next_target = jnp.min(q_values, axis=0)
                next_q_value = jax.lax.stop_gradient(
                    data.rewards + (1 - data.dones) * self.gamma * min_qf_next_target
                )

                q_pred = self.critic.apply_fn(params, data.observations, data.actions)
                loss = 0.5 * ((q_pred - next_q_value) ** 2).mean(axis=1).sum()
                return loss, q_pred.mean()

            (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(
                critic_loss, has_aux=True
            )(_critic.params)
            _critic = _critic.apply_gradients(grads=critic_grads)
            flat_grads, _ = flatten_util.ravel_pytree(critic_grads)
            return _critic, {
                "losses/qf_values": qf_values,
                "losses/qf_loss": critic_loss_value,
                "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            }

        # Update critic
        alpha_val = self.alpha.apply_fn(self.alpha.params)
        critic, critic_logs = update_critic(self.critic, alpha_val)

        # Update diffusion model
        def diffusion_loss(params: FrozenDict) -> tuple[Float[Array, ""], LogDict]:
            batch_size = data.observations.shape[0]
            
            # Sample random timesteps
            t = jax.random.uniform(
                diffusion_loss_key, (batch_size,), minval=0.0, maxval=1.0
            )
            
            # Sample noise
            noise = jax.random.normal(diffusion_loss_key, data.actions.shape)
            
            # Add noise to actions
            alpha_t, _ = ddpm_schedule(t * self.diffusion_schedule_steps, self.diffusion_schedule_steps)
            alpha_t = alpha_t[..., None]
            noisy_actions = jnp.sqrt(alpha_t) * data.actions + jnp.sqrt(1 - alpha_t) * noise
            
            # Predict noise
            predicted_noise = self.diffusion_net.apply_fn(
                params, noisy_actions, t, data.observations
            )
            
            # Compute MSE loss for noise prediction
            mse_loss = ((predicted_noise - noise) ** 2).mean()
            
            # Add policy gradient term (guidance)
            q_values = critic.apply_fn(critic.params, data.observations, data.actions)
            min_q = jnp.min(q_values, axis=0)
            policy_loss = -min_q.mean()
            
            total_loss = mse_loss + alpha_val.mean() * policy_loss
            
            return total_loss, {
                "losses/diffusion_mse": mse_loss,
                "losses/diffusion_policy": policy_loss,
                "losses/diffusion_total": total_loss,
            }

        (diffusion_loss_value, diffusion_logs), diffusion_grads = jax.value_and_grad(
            diffusion_loss, has_aux=True
        )(self.diffusion_net.params)
        
        diffusion_net = self.diffusion_net.apply_gradients(grads=diffusion_grads)
        flat_grads, _ = flatten_util.ravel_pytree(diffusion_grads)
        diffusion_logs["metrics/diffusion_grad_magnitude"] = jnp.linalg.norm(flat_grads)

        # Update temperature (same as SAC)
        def alpha_loss(params: FrozenDict) -> Float[Array, ""]:
            # Estimate entropy of current policy
            sampled_actions, _ = _sample_action_ddpm(
                diffusion_net, data.observations, diffusion_loss_key, self.num_diffusion_steps
            )
            # Use approximate entropy estimation
            log_alpha: jax.Array
            log_alpha = params["params"]["log_alpha"]
            entropy_estimate = 0.0  # Simplified - could use more sophisticated estimation
            return -log_alpha * (entropy_estimate + self.target_entropy)

        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(self.alpha.params)
        alpha = self.alpha.apply_gradients(grads=alpha_grads)
        
        # Update target critic
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params,
                critic.target_params,
                self.tau,
            )
        )

        self = self.replace(
            key=key,
            diffusion_net=diffusion_net,
            critic=critic,
            alpha=alpha,
        )

        logs = {
            **critic_logs,
            **diffusion_logs,
            "losses/alpha_loss": alpha_loss_value,
            "alpha": jnp.exp(alpha.params["params"]["log_alpha"]).sum(),
        }

        return self, logs

    @override
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        return self._update_inner(data)

    def _split_critic_activations(
        self, critic_acts: LayerActivationsDict
    ) -> tuple[LayerActivationsDict, ...]:
        return tuple(
            {key: value[i] for key, value in critic_acts.items()}
            for i in range(self.num_critics)
        )

    @jax.jit
    def _get_intermediates(
        self, data: ReplayBufferSamples
    ) -> tuple[Self, Intermediates, Intermediates]:
        key, activations_key = jax.random.split(self.key, 2)

        batch_size = data.observations.shape[0]
        
        # Get diffusion network activations
        dummy_time = jnp.ones((batch_size,))
        _, diffusion_state = self.diffusion_net.apply_fn(
            self.diffusion_net.params, data.actions, dummy_time, data.observations, mutable="intermediates"
        )
        
        # Get critic activations
        _, critic_state = self.critic.apply_fn(
            self.critic.params, data.observations, data.actions, mutable="intermediates"
        )

        diffusion_intermediates = jax.tree.map(
            lambda x: x.reshape(batch_size, -1), diffusion_state["intermediates"]
        )
        critic_intermediates = jax.tree.map(
            lambda x: x.reshape(self.num_critics, batch_size, -1),
            critic_state["intermediates"]["VmapQValueFunction_0"],
        )

        self = self.replace(key=key)

        return (
            self,
            diffusion_intermediates,
            critic_intermediates,
        )