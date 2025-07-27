import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Callable

from metaworld_algorithms.config.nn import DIMEConfig


class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: Float[Array, "batch"]) -> Float[Array, "batch dim"]:
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[..., None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class MLPBlock(nn.Module):
    features: int
    activation: Callable = nn.relu
    use_batch_norm: bool = False

    @nn.compact
    def __call__(self, x: Float[Array, "batch features"]) -> Float[Array, "batch features"]:
        x = nn.Dense(self.features)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not self.is_training())(x)
        x = self.activation(x)
        return x


class DIMENetwork(nn.Module):
    config: DIMEConfig

    action_dim: int
    hidden_dims: tuple[int, ...] = (256, 256, 256) 
    time_embed_dim: int = 128
    cond_embed_dim: int = 128
    activation: Callable = nn.gelu # TODO: compare with nn.silu

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "batch action_dim"],
        time: Float[Array, "batch"],
        cond: Float[Array, "batch obs_dim"],
    ) -> Float[Array, "batch action_dim"]:
        # Time embedding
        time_emb = SinusoidalPosEmb(self.time_embed_dim)(time)
        time_emb = nn.Dense(self.time_embed_dim)(time_emb)
        time_emb = self.activation(time_emb)
        time_emb = nn.Dense(self.time_embed_dim)(time_emb)

        # Condition embedding
        cond_emb = nn.Dense(self.cond_embed_dim)(cond)
        cond_emb = self.activation(cond_emb)

        # Main network
        h = jnp.concatenate([x, time_emb, cond_emb], axis=-1)
        
        for hidden_dim in self.hidden_dims:
            h = MLPBlock(hidden_dim, self.activation)(h)
        
        # Output layer
        output = nn.Dense(self.action_dim)(h)
        return output