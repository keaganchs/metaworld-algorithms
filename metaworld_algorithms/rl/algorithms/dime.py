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


class ScoreNetwork(nn.Module):
    """Score network for DIME diffusion policy."""
    
    action_dim: int
    hidden_dim: int = 256
    num_layers: int = 3

    @nn.compact
    def __call__(self, 
                 action: Action, 
                 observation: Observation, 
                 timestep: jnp.ndarray,
                 langevin_vals: jnp.ndarray) -> Action:
        """Predict score (gradient of log probability) for denoising."""
        batch_shape = observation.shape[:-1]
        
        # Convert timestep to proper shape for broadcasting
        timestep_array = jnp.full(batch_shape + (1,), timestep, dtype=jnp.float32)
        
        # Concatenate inputs: action, observation, timestep, langevin values
        x = jnp.concatenate([action, observation, timestep_array, langevin_vals], axis=-1)

        # Forward pass through the network
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
        
        # Final output layer - predict score (same dimension as action)
        score = nn.Dense(self.action_dim)(x)
        return score


class DiffusionPolicy(nn.Module):
    """DIME-style diffusion policy with optimal dynamics."""
    
    action_dim: int
    num_diffusion_steps: int
    hidden_dim: int = 256
    num_layers: int = 3
    initial_std: float = 1.0
    friction: float = 1.0
    
    def setup(self):
        # Initialize diffusion parameters
        self.score_network = ScoreNetwork(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Learnable parameters for diffusion process
        self.prior_mean = self.param(
            "prior_mean", 
            init_fn=lambda _: jnp.zeros(self.action_dim)
        )
        self.prior_log_std = self.param(
            "prior_log_std",
            init_fn=lambda _: jnp.full(self.action_dim, jnp.log(self.initial_std))
        )
        self.dt = self.param(
            "dt",
            init_fn=lambda _: jnp.full(self.num_diffusion_steps, 0.1)
        )
        self.log_friction = self.param(
            "log_friction",
            init_fn=lambda _: jnp.full(self.action_dim, jnp.log(self.friction))
        )
    
    def __call__(self, method_name: str, *args, **kwargs):
        """Method dispatcher for different diffusion operations."""
        methods = {
            "prior_sampler": self.prior_sampler,
            "prior_log_prob": self.prior_log_prob,
            "delta_t_fn": self.delta_t_fn,
            "friction_fn": self.friction_fn,
            "forward_model": self.forward_model,
            "drift_fn": self.drift_fn
        }

        if method_name in methods:
            return methods[method_name](*args, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def prior_sampler(self, key: PRNGKeyArray, n_samples: int) -> jnp.ndarray:
        """Sample from prior distribution."""
        std = jnp.exp(self.prior_log_std)
        return jax.random.normal(key, (n_samples, self.action_dim)) * std + self.prior_mean
    
    def prior_log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute prior log probability."""
        std = jnp.exp(self.prior_log_std)
        dist = distrax.MultivariateNormalDiag(self.prior_mean, std)
        return dist.log_prob(x)
    
    def delta_t_fn(self, step: int) -> float:
        """Get timestep delta for given step."""
        return jax.nn.softplus(self.dt[step])
    
    def friction_fn(self) -> jnp.ndarray:
        """Get friction coefficient."""
        return jax.nn.softplus(self.log_friction)
    
    def forward_model(self, step: int, x: jnp.ndarray, obs: jnp.ndarray, 
                     langevin_vals: jnp.ndarray) -> jnp.ndarray:
        """Forward model for diffusion process."""
        return self.score_network(x, obs, step, langevin_vals)
    
    def drift_fn(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute drift term (score of prior)."""
        return jax.grad(self.prior_log_prob)(x)


@partial(jax.jit, static_argnames=["action_dim", "num_steps"])
def diffusion_sample_single(
    actor_params, observation: Observation, key: PRNGKeyArray,
    action_dim: int, num_steps: int
) -> tuple[Action, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample single action using DIME-style diffusion."""
    # Extract single observation for non-batched processing
    if observation.ndim > 1:
        obs = observation[0]  # Take first element if batched
    else:
        obs = observation
    
    # For now, implement simplified diffusion sampling
    # Sample from a simple normal distribution and apply tanh
    key, prior_key = jax.random.split(key)
    init_x = jax.random.normal(prior_key, (action_dim,))
    
    # Apply tanh squashing for bounded actions
    final_action = jnp.tanh(init_x)
    
    # Placeholder costs - to be properly implemented with full SDE integration
    running_costs = jnp.array(0.0)
    stochastic_costs = jnp.array(0.0) 
    terminal_costs = jnp.array(0.0)
    
    return final_action, running_costs, stochastic_costs, terminal_costs


@partial(jax.jit, static_argnames=["action_dim", "num_steps"])
def diffusion_sample(
    actor_params, observation: Observation, key: PRNGKeyArray,
    action_dim: int, num_steps: int
) -> tuple[Action, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Efficient DIME-style diffusion sampling for batched inputs."""
    batch_shape = observation.shape[:-1]
    
    if len(batch_shape) == 0:
        # Single observation
        return diffusion_sample_single(actor_params, observation, key, action_dim, num_steps)
    
    # Split keys for each batch element
    keys = jax.random.split(key, num=batch_shape[0])
    
    # Vectorized sampling
    def single_sample_fn(key_obs_pair):
        k, obs = key_obs_pair
        return diffusion_sample_single(actor_params, obs, k, action_dim, num_steps)
    
    # Map over batch
    results = jax.vmap(single_sample_fn)((keys, observation))
    actions, running_costs, stochastic_costs, terminal_costs = results
    
    return actions, running_costs, stochastic_costs, terminal_costs

@partial(jax.jit, static_argnames=("action_dim", "num_steps"))
def _sample_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray,
    action_dim: int, num_steps: int
) -> tuple[Float[Array, "... action_dim"], jnp.ndarray, jnp.ndarray, jnp.ndarray, PRNGKeyArray]:
    action_key, next_key = jax.random.split(key)
    action, running_costs, stochastic_costs, terminal_costs = diffusion_sample(
        actor.params, observation, action_key, action_dim, num_steps
    )
    return action, running_costs, stochastic_costs, terminal_costs, next_key

@partial(jax.jit, static_argnames=("action_dim", "num_steps"))
def _eval_action(
    actor: TrainState, observation: Observation, key: PRNGKeyArray,
    action_dim: int, num_steps: int
) -> Float[Array, "... action_dim"]:
    action, _, _, _ = diffusion_sample(
        actor.params, observation, key, action_dim, num_steps
    )
    return action


def extract_task_weights(
    alpha_params: FrozenDict, task_ids: Float[np.ndarray, "... num_tasks"]
) -> Float[Array, "... 1"]:
    log_alpha: jax.Array
    task_weights: jax.Array

    log_alpha = alpha_params["params"]["log_alpha"]  # pyright: ignore [reportAssignmentType]
    task_weights = jax.nn.softmax(-log_alpha)
    task_weights = task_ids @ task_weights.reshape(-1, 1)  # pyright: ignore [reportAssignmentType]
    task_weights *= log_alpha.shape[0]
    return task_weights


# OK
@dataclasses.dataclass(frozen=True)
class DIMEConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig(
        use_classification=True,  # Enable distributional Q-learning
    )
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    policy_tau: float = 0.005 # TODO
    use_task_weights: bool = False
    num_diffusion_steps: int = 16
    diffusion_hidden_dim: int = 256
    diffusion_num_layers: int = 3
    policy_delay: int = 2
    entropy_coefficient: float = 0.1
    logging_frequency: int = 1000  # How often to log metrics (reduced for performance)
    
    # Distributional Q-learning parameters
    v_min: float = -10.0
    v_max: float = 10.0
    num_atoms: int = 51


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

    # Distributional Q-learning parameters
    v_min: float = struct.field(pytree_node=False)
    v_max: float = struct.field(pytree_node=False)
    num_atoms: int = struct.field(pytree_node=False)

    # Training configuration
    use_task_weights: bool = struct.field(pytree_node=False)
    split_actor_losses: bool = struct.field(pytree_node=False)
    split_critic_losses: bool = struct.field(pytree_node=False)
    
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
        dummy_key = jax.random.PRNGKey(0)
        
        action_dim = int(np.prod(env_config.action_space.shape))

        # Initialize actor (diffusion policy)
        actor_net = DiffusionPolicy(
            action_dim=action_dim,
            num_diffusion_steps=config.num_diffusion_steps,
            hidden_dim=config.diffusion_hidden_dim,
            num_layers=config.diffusion_num_layers,
        )
        actor_params = actor_net.init(actor_key, "prior_sampler", dummy_key, 1)
        actor = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_params,
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
            v_min=config.v_min,
            v_max=config.v_max,
            num_atoms=config.num_atoms,
            use_task_weights=config.use_task_weights,
            split_actor_losses=config.actor_config.network_config.optimizer.requires_split_task_losses,
            split_critic_losses=config.critic_config.network_config.optimizer.requires_split_task_losses,
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

    def split_data_by_tasks(
        self,
        data: PyTree[Float[Array, "batch data_dim"]],
        task_ids: Float[npt.NDArray, "batch num_tasks"],
    ) -> PyTree[Float[Array, "num_tasks per_task_batch data_dim"]]:
        tasks = jnp.argmax(task_ids, axis=1)
        sorted_indices = jnp.argsort(tasks)

        def group_by_task_leaf(
            leaf: Float[Array, "batch data_dim"],
        ) -> Float[Array, "task task_batch data_dim"]:
            leaf_sorted = leaf[sorted_indices]
            return leaf_sorted.reshape(self.num_tasks, -1, leaf.shape[1])

        return jax.tree.map(group_by_task_leaf, data), sorted_indices

    def unsplit_data_by_tasks(
        self,
        split_data: PyTree[Float[Array, "num_tasks per_task_batch data_dim"]],
        sort_indices: jax.Array,
    ) -> PyTree[Float[Array, "batch data_dim"]]:
        def reconstruct_leaf(
            leaf: Float[Array, "num_tasks per_task_batch data_dim"],
        ) -> Float[Array, "batch data_dim"]:
            batch_size = leaf.shape[0] * leaf.shape[1]
            flat = leaf.reshape(batch_size, leaf.shape[-1])
            # Create inverse permutation
            inverse_indices = jnp.zeros_like(sort_indices)
            inverse_indices = inverse_indices.at[sort_indices].set(
                jnp.arange(batch_size)
            )
            return flat[inverse_indices]

        return jax.tree.map(reconstruct_leaf, split_data)

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        action, running_costs, stochastic_costs, terminal_costs, key = _sample_action(
            self.actor, observation, self.key, self.action_dim, self.num_diffusion_steps
        )
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observation: Observation) -> Action:
        return jax.device_get(_eval_action(
            self.actor, observation, self.key, self.action_dim, self.num_diffusion_steps
        ))
    
    def update_critic(
        self,
        data: ReplayBufferSamples,
        temperature_val: Float[Array, "*batch 1"],
        task_weights: Float[Array, "*batch 1"] | None = None,
    ) -> tuple[Self, LogDict]:
        """Update critic networks with distributional Q-learning."""
        key, critic_loss_key = jax.random.split(self.key)
        
        # Create atom support for distributional Q-learning
        v_min, v_max, num_atoms = self.v_min, self.v_max, self.num_atoms
        z_atoms = jnp.linspace(v_min, v_max, num_atoms)
        
        def categorical_projection(next_dist, rewards, dones, gamma, v_min, v_max, num_atoms, support):
            """Project target distribution onto atom support."""
            delta_z = (v_max - v_min) / (num_atoms - 1)
            batch_size = rewards.shape[0]
            
            # Expand support for batch processing
            support = jnp.expand_dims(support, 0)  # [1, num_atoms]
            
            target_z = jnp.clip(
                rewards[:, None] + (1 - dones[:, None]) * gamma * support,
                a_min=v_min, a_max=v_max
            )
            
            # Compute indices for distribution projection
            b = (target_z - v_min) / delta_z
            l = jnp.floor(b).astype(jnp.int32)
            u = jnp.ceil(b).astype(jnp.int32)
            
            # Handle edge cases where l == u  
            u = jnp.where((l < (num_atoms - 1)) & (l == u), u + 1, u)
            
            # Initialize projection distribution
            proj_dist = jnp.zeros((batch_size, num_atoms))
            
            # Compute projection weights for lower indices
            l_mask = (l >= 0) & (l < num_atoms)
            l_weights = next_dist * (u.astype(jnp.float32) - b)
            
            # Compute projection weights for upper indices  
            u_mask = (u >= 0) & (u < num_atoms) 
            u_weights = next_dist * (b - l.astype(jnp.float32))
            
            # Add weights to projection distribution
            proj_dist = proj_dist.at[jnp.arange(batch_size)[:, None], l].add(
                jnp.where(l_mask, l_weights, 0.0)
            )
            proj_dist = proj_dist.at[jnp.arange(batch_size)[:, None], u].add(
                jnp.where(u_mask, u_weights, 0.0)
            )
            
            return proj_dist

        def critic_loss_fn(critic_params):
            # Current Q-distributions
            q_distributions = self.critic.apply_fn(
                critic_params, data.observations, data.actions
            )
            
            # Sample next actions using diffusion sampling
            next_actions, _, _, _ = diffusion_sample(
                self.actor.params, data.next_observations, critic_loss_key,
                self.action_dim, self.num_diffusion_steps
            )
            
            target_q_distributions = self.critic.apply_fn(
                self.critic.target_params, data.next_observations, next_actions
            )
            
            num_critics = len(target_q_distributions)
            
            # Project target distributions for all critics
            target_projections = []
            for i in range(num_critics):
                target_proj = categorical_projection(
                    target_q_distributions[i], data.rewards, data.dones, self.gamma,
                    v_min, v_max, num_atoms, z_atoms
                )
                target_projections.append(target_proj)
            
            # Average the projected targets across all critics
            target_dist = jax.lax.stop_gradient(
                jnp.mean(jnp.stack(target_projections, axis=0), axis=0)
            )
            
            # Cross-entropy loss for distributional Q-learning
            def cross_entropy_loss(pred_dist, target_dist):
                return -jnp.mean(jnp.sum(target_dist * jnp.log(pred_dist + 1e-15), axis=-1))
            
            # Compute losses for all critics
            critic_losses = []
            q_values_list = []
            entropy_list = []
            
            for i in range(num_critics):
                current_q_dist = q_distributions[i]
                loss = cross_entropy_loss(current_q_dist, target_dist)
                critic_losses.append(loss)
                
                # Compute Q-values from distribution for logging
                q_values = jnp.sum(current_q_dist * z_atoms, axis=-1)
                q_values_list.append(q_values)
                
                # Compute entropy for logging
                entropy = -jnp.mean(jnp.sum(current_q_dist * jnp.log(current_q_dist + 1e-15), axis=-1))
                entropy_list.append(entropy)
            
            total_loss = jnp.sum(jnp.stack(critic_losses))
            
            # Take minimum Q-values across all critics for conservative estimate
            min_q_values = jnp.min(jnp.stack(q_values_list, axis=0), axis=0)
            
            # Build logs dictionary
            logs = {
                "losses/critic_loss": total_loss,
                "losses/critic_q_values": jnp.mean(min_q_values),
            }
            
            # Add individual critic logs
            for i in range(num_critics):
                logs[f"losses/critic_q{i+1}_loss"] = critic_losses[i]
                logs[f"metrics/critic_entropy_q{i+1}"] = entropy_list[i]
            
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
        flat_params_crit, _ = flatten_util.ravel_pytree(new_critic.params)

        return self.replace(critic=new_critic, key=key), {
            **logs, 
            "metrics/critic_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/critic_params_norm": jnp.linalg.norm(flat_params_crit),
        }    # Alpha
    @jax.jit
    def update_temperature( 
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
            "losses/temperature_loss": temperature_loss_value,
            "temperature": jnp.exp(temperature.params["params"]["log_alpha"]).sum(),  # pyright: ignore [reportArgumentType]
        }

    def update_actor(
        self,
        data: ReplayBufferSamples,
        temperature_val: Float[Array, "batch 1"],
        task_weights: Float[Array, "batch 1"] | None = None,
    ) -> tuple[Self, Float[Array, " batch"], LogDict]:
        """Update actor (diffusion policy) with distributional Q-values."""
        key, actor_loss_key = jax.random.split(self.key)
        
        # Distributional Q parameters (same as in critic)
        v_min, v_max, num_atoms = self.v_min, self.v_max, self.num_atoms
        z_atoms = jnp.linspace(v_min, v_max, num_atoms)
        
        def actor_loss_fn(actor_params):
            # Sample actions from diffusion policy with entropy costs
            actions, running_costs, stochastic_costs, terminal_costs = diffusion_sample(
                actor_params, data.observations, actor_loss_key,
                self.action_dim, self.num_diffusion_steps
            )
            
            # Q-distributions
            q_distributions = self.critic.apply_fn(
                self.critic.params, data.observations, actions
            )
            
            num_critics = len(q_distributions)
            
            # Convert distributions to Q-values for all critics
            q_values_list = []
            for i in range(num_critics):
                q_values = jnp.sum(q_distributions[i] * z_atoms, axis=-1)
                q_values_list.append(q_values)
            
            # Take minimum across all critics for conservative estimate
            min_q_values = jnp.min(jnp.stack(q_values_list, axis=0), axis=0)

            # DIME-style actor loss with entropy costs
            total_entropy_costs = running_costs + stochastic_costs + terminal_costs
            if task_weights is not None:
                actor_loss = (task_weights.squeeze() * (-min_q_values + temperature_val.squeeze() * total_entropy_costs)).mean()
            else:
                actor_loss = (-min_q_values + temperature_val.squeeze() * total_entropy_costs).mean()
            
            return actor_loss, total_entropy_costs
        
        # Update actor
        (actor_loss_value, log_probs), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(self.actor.params)
        
        new_actor = self.actor.apply_gradients(grads=actor_grads)
        
        # Compute gradient and parameter norms for logging (similar to MTSAC)
        flat_grads, _ = flatten_util.ravel_pytree(actor_grads)
        flat_params_act, _ = flatten_util.ravel_pytree(new_actor.params)
        
        logs = {
            "losses/actor_loss": actor_loss_value,
            "metrics/actor_grad_magnitude": jnp.linalg.norm(flat_grads),
            "metrics/actor_params_norm": jnp.linalg.norm(flat_params_act),
        }
        
        return self.replace(actor=new_actor, key=key), log_probs, logs

    @jax.jit
    def _update_inner(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        """Update the DIME algorithm following MTSAC structure."""
        task_ids = data.observations[..., -self.num_tasks :]

        temperature_vals = self.temperature.apply_fn(self.temperature.params, task_ids)
        if self.use_task_weights:
            task_weights = extract_task_weights(self.temperature.params, task_ids)
        else:
            task_weights = None

        actor_data = critic_data = data
        actor_temperature_vals = critic_temperature_vals = temperature_vals
        actor_task_weights = critic_task_weights = task_weights
        temperature_val_indices = None

        if self.split_critic_losses or self.split_actor_losses:
            split_data, _ = self.split_data_by_tasks(data, task_ids)
            split_temperature_vals, temperature_val_indices = self.split_data_by_tasks(
                temperature_vals, task_ids
            )
            split_task_weights, _ = (
                self.split_data_by_tasks(task_weights, task_ids)
                if task_weights is not None
                else (None, None)
            )

            if self.split_critic_losses:
                critic_data = split_data
                critic_temperature_vals = split_temperature_vals
                critic_task_weights = split_task_weights

            if self.split_actor_losses:
                actor_data = split_data
                actor_temperature_vals = split_temperature_vals
                actor_task_weights = split_task_weights

        self, critic_logs = self.update_critic(
            critic_data, critic_temperature_vals, critic_task_weights
        )
        self, log_probs, actor_logs = self.update_actor(
            actor_data, actor_temperature_vals, actor_task_weights
        )
        if self.split_actor_losses:
            assert temperature_val_indices is not None
            log_probs = self.unsplit_data_by_tasks(log_probs, temperature_val_indices)
        self, temperature_logs = self.update_temperature(log_probs, task_ids)

        return self, {
            **critic_logs,
            **actor_logs,
            **temperature_logs,
        }

    @override
    def update(self, batch: ReplayBufferSamples) -> tuple[Self, LogDict]:
        return self._update_inner(batch)
