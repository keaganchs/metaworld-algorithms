from dataclasses import dataclass

from .optim import OptimizerConfig
from .utils import Activation, CellType, Initializer
# from .networks import QValueFunctionConfig

@dataclass(frozen=True)
class NetworkConfig:
    """Base config for neural networks."""
    
    optimizer: OptimizerConfig = OptimizerConfig()



@dataclass(frozen=True, kw_only=True)
class NeuralNetworkConfig(NetworkConfig):
    width: int = 400
    """The number of neurons in the hidden layers."""

    depth: int = 3
    """The number of hidden layers."""

    kernel_init: Initializer = Initializer.HE_UNIFORM
    """The initializer to use for hidden layer weights."""
    # TODO: How to pass arguments to kernel_init

    bias_init: Initializer = Initializer.ZEROS
    """The initializer to use for hidden layer biases."""
    # TODO: How to pass arguments to bias_init

    use_bias: bool = True
    """Whether or not to use bias terms across the network."""

    activation: Activation = Activation.ReLU
    """The activation function to use."""


@dataclass(frozen=True, kw_only=True)
class RecurrentNeuralNetworkConfig:
    width: int = 256
    """The dimension of the recurrent layer and the hidden state."""

    cell_type: CellType = CellType.GRU
    """The type of recurrent cell to use."""

    recurrent_kernel_init: Initializer = Initializer.HE_UNIFORM
    """The initializer to use for the recurrent layer weights."""

    kernel_init: Initializer = Initializer.HE_UNIFORM
    """The initializer to use for the recurrent layer weights."""

    bias_init: Initializer = Initializer.ZEROS
    """The initializer to use for the recurrent layer biases."""

    use_bias: bool = True
    """Whether or not to use bias terms across the network."""

    carry_init: Initializer = Initializer.ZEROS
    """The initializer to use for the recurrent carry."""

    activation: Activation = Activation.ReLU
    """The activation function to use after the recurrent layer."""

    optimizer: OptimizerConfig = OptimizerConfig()
    """The optimizer to use for the whole network."""


@dataclass(frozen=True, kw_only=True)
class VanillaNetworkConfig(NeuralNetworkConfig):
    use_skip_connections: bool = False
    """Whether or not to use skip connections."""

    use_layer_norm: bool = False
    """Whether or not to use layer normalization."""


@dataclass(frozen=True, kw_only=True)
class MultiHeadConfig(NeuralNetworkConfig):
    num_tasks: int
    """The number of tasks (used for extracting the task IDs & to determine the number of heads)."""


@dataclass(frozen=True, kw_only=True)
class SoftModulesConfig(NeuralNetworkConfig):
    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    width: int = 256
    """The number of neurons in the Dense layers around the network."""

    module_width: int = 256
    """The number of neurons in each module in the Base Policy Network. `d` in the paper."""

    depth: int = 2
    """The number of Base Policy Network modules layers."""
    # 2 for MT10, 4 for MT50

    num_modules: int = 2
    """The number of modules to use in each Base Policy Network layer."""
    # 2 for MT10, 4 for MT50

    embedding_dim: int = 400
    """The dimension of the observation / task index embedding. `D` in the paper."""


@dataclass(frozen=True, kw_only=True)
class PaCoConfig(NeuralNetworkConfig):
    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    num_parameter_sets: int = 5
    """The number of parameter sets. `K` in the paper."""


@dataclass(frozen=True, kw_only=True)
class CAREConfig(NeuralNetworkConfig):
    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    encoder_width: int = 50
    """The width of the Context Encoder network."""

    encoder_depth: int = 2
    """The depth of the Context Encoder network."""

    encoder_temperature: float = 1.0
    """The temperature of the encoder attention."""

    embedding_dim: int = 50
    """The dimensionality of the context embedding."""

    num_experts: int = 6
    """The number of experts in CARE's task-dependent MoE observation encoder."""
    # 6 for MT10, 10 for MT50


@dataclass(frozen=True, kw_only=True)
class FiLMConfig(NeuralNetworkConfig):
    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    encoder_width: int = 50
    """The width of the Context Encoder network."""

    encoder_depth: int = 2
    """The depth of the Context Encoder network."""

    embedding_dim: int = 50
    """The dimensionality of the context embedding."""


@dataclass(frozen=True, kw_only=True)
class MOOREConfig(NeuralNetworkConfig):
    num_tasks: int
    """The number of tasks (used for extracting the task IDs)."""

    num_experts: int = 4
    """The number of orthogonal experts."""
    # Original values are 4 for MT10 and 6 for MT50


@dataclass(frozen=True, kw_only=True)
class DIMEConfig(VanillaNetworkConfig):
    diffusion_config: VanillaNetworkConfig = VanillaNetworkConfig()
    """The configuration for the diffusion network, which is used to generate task-specific policies."""

    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    """The configuration for the temperature optimizer, which is used to adjust the exploration-exploitation trade-off."""

    initial_temperature: float = 1.0
    """The initial temperature controls the exploration-exploitation trade-off."""

    num_critics: int = 2
    """The number of critics."""

    tau: float = 0.005
    """The rate at which the target network is updated."""

    num_diffusion_steps: int = 16
    """The number of diffusion steps to take during training. This controls the number of policy updates per training step.
    16 was found to be optimal in the original DIME paper, but 8 or 4 steps can also be used for faster training and inference."""
    
    diffusion_schedule_steps: int = 1000
    """The number of steps for the diffusion schedule, which controls the noise level in the diffusion process."""

    ema_rate: float = 0.995
    """The rate at which the exponential moving average of the policy is updated.
    This is used to stabilize the training process by smoothing out the policy updates."""
