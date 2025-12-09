"""
Math RL Training Recipe - Arithmetic, GSM8K, MATH datasets
Supports training models on mathematical reasoning tasks with RL
"""

import asyncio
import chz
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl import arithmetic_env, math_env
from tinker_cookbook.rl.train import Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder


@chz.chz
class MathRLConfig:
    """Configuration for Math RL training"""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Environment: arithmetic, math, gsm8k, polaris, deepmath
    env: str = "arithmetic"

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 256

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None

    # Checkpointing
    eval_every: int = 20
    save_every: int = 20


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
) -> RLDatasetBuilder:
    """Create dataset builder for specified environment"""
    if env == "arithmetic":
        return arithmetic_env.ArithmeticDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            n_batches=100,
            include_fewshot=True,
            group_size=group_size,
        )
    elif env in ["math", "polaris", "deepmath", "gsm8k"]:
        return math_env.get_math_dataset_builder(
            dataset_name=env,
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=0,
        )
    else:
        raise ValueError(f"Unknown math environment: {env}")


def build_config_blueprint(user_config: dict) -> chz.Blueprint[Config]:
    """Build training config from user parameters"""

    # Extract config
    model_name = user_config.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
    env_name = user_config.get("environment", "arithmetic")
    hyperparams = user_config.get("hyperparameters", {})

    # Get hyperparameters with defaults
    lora_rank = hyperparams.get("rank", 32)
    learning_rate = hyperparams.get("learning_rate", 1e-5)
    group_size = hyperparams.get("group_size", 4)
    groups_per_batch = hyperparams.get("groups_per_batch", 100)
    max_tokens = hyperparams.get("max_tokens", 256)

    # Get renderer
    renderer_name = user_config.get(
        "renderer_name"
    ) or model_info.get_recommended_renderer_name(model_name)

    # Create dataset builder
    dataset_builder = get_dataset_builder(
        env=env_name,
        batch_size=groups_per_batch,
        model_name=model_name,
        renderer_name=renderer_name,
        group_size=group_size,
    )

    # Log path from config or default
    log_path = user_config.get("log_path", f"/tmp/tinker-platform/math_rl/{env_name}")

    return chz.Blueprint(Config).apply(
        {
            "model_name": model_name,
            "lora_rank": lora_rank,
            "dataset_builder": dataset_builder,
            "learning_rate": learning_rate,
            "max_tokens": max_tokens,
            "log_path": log_path,
            "eval_every": hyperparams.get("eval_every", 20),
            "save_every": hyperparams.get("save_every", 20),
            "wandb_project": user_config.get("wandb_project"),
        }
    )


async def run_math_rl(user_config: dict, log_callback=None):
    """Run Math RL training with user configuration"""
    try:
        blueprint = build_config_blueprint(user_config)
        config = blueprint.make()

        # Check log directory
        cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")

        if log_callback:
            log_callback(
                f"Starting Math RL training on {user_config.get('environment', 'arithmetic')} environment..."
            )

        # Run training
        await main(config)

        if log_callback:
            log_callback(f"Math RL training completed! Logs saved to {config.log_path}")

    except Exception as e:
        if log_callback:
            log_callback(f"Error in Math RL training: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage for testing
    test_config = {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "environment": "arithmetic",
        "hyperparameters": {
            "learning_rate": 1e-5,
            "rank": 32,
            "group_size": 4,
            "groups_per_batch": 100,
            "max_tokens": 256,
        },
    }
    asyncio.run(run_math_rl(test_config))
