from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.nn import VanillaNetworkConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OffPolicyTrainingConfig
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.rl.algorithms import DIMEConfig
from metaworld_algorithms.run import Run


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_notes: str | None = None
    data_dir: Path = Path("./run_results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)

    run = Run(
        run_name="mt10_dime",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
        ),
        algorithm=DIMEConfig(
            num_tasks=10,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=VanillaNetworkConfig(
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            num_critics=2,
            initial_temperature=1.0,
            tau=0.005,
            policy_tau=0.005,
            num_diffusion_steps=4,  # Reduced for faster testing from 16
            diffusion_hidden_dim=256,
            diffusion_num_layers=3,
            policy_delay=2,
            entropy_coefficient=0.1,
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(2e7),
            buffer_size=int(1e6),
            batch_size=640, # Reduced for faster testing from 1280
        ),
        checkpoint=True,
        resume=args.resume,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
        run.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            notes=args.wandb_notes if hasattr(args, 'wandb_notes') else None,
            config=run,
            resume="allow",
        )

    run.start()


if __name__ == "__main__":
    main()
