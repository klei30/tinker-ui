"""
Hyperparameter calculator using tinker-cookbook implementation.

References:
- https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams
- https://tinker-docs.thinkingmachines.ai/lora-primer
- tinker-cookbook/tinker_cookbook/hyperparam_utils.py

Formula for learning rate:
LR(m) = lr_base · M_LoRA · (2000/H_m)^P_m

Where:
- lr_base = 5e-5 (base learning rate constant)
- M_LoRA = 10 (multiplier for LoRA; 1 for full fine-tuning)
- H_m = model's hidden size
- P_m = model-specific exponent (0.0775 for Qwen; 0.781 for Llama)
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class HyperparamCalculator:
    """Calculate recommended hyperparameters based on tinker-cookbook implementation."""

    # Model hidden sizes - directly from tinker-cookbook
    HIDDEN_SIZES = {
        # Llama models
        "meta-llama/Llama-3.2-1B": 2048,
        "meta-llama/Llama-3.2-1B-Instruct": 2048,
        "meta-llama/Llama-3.2-3B": 3072,
        "meta-llama/Llama-3.2-3B-Instruct": 3072,
        "meta-llama/Llama-3.1-8B": 4096,
        "meta-llama/Llama-3.1-8B-Instruct": 4096,
        "meta-llama/Llama-3.1-70B": 8192,
        "meta-llama/Llama-3.3-70B-Instruct": 8192,
        # Qwen models
        "Qwen/Qwen2.5-0.5B": 896,
        "Qwen/Qwen2.5-0.5B-Instruct": 896,
        "Qwen/Qwen2.5-1.5B": 1536,
        "Qwen/Qwen2.5-1.5B-Instruct": 1536,
        "Qwen/Qwen2.5-3B": 2048,
        "Qwen/Qwen2.5-3B-Instruct": 2048,
        "Qwen/Qwen2.5-7B": 3584,
        "Qwen/Qwen2.5-7B-Instruct": 3584,
        "Qwen/Qwen2.5-14B": 5120,
        "Qwen/Qwen2.5-14B-Instruct": 5120,
        "Qwen/Qwen2.5-32B": 5120,
        "Qwen/Qwen2.5-32B-Instruct": 5120,
        "Qwen/Qwen2.5-72B": 8192,
        "Qwen/Qwen2.5-72B-Instruct": 8192,
    }

    # Model-specific exponents - directly from tinker-cookbook
    MODEL_EXPONENTS = {
        "llama": 0.781,
        "qwen": 0.0775,
    }

    # Constants from tinker-cookbook
    LR_BASE = 5e-5
    LORA_MULTIPLIER = 10.0  # From get_lora_lr_over_full_finetune_lr()

    @staticmethod
    def _get_hidden_size(model_name: str) -> int:
        """Get the hidden size for a model - matching tinker-cookbook implementation."""
        if model_name in HyperparamCalculator.HIDDEN_SIZES:
            return HyperparamCalculator.HIDDEN_SIZES[model_name]

        # Default for unknown models
        logger.warning(
            f"Model {model_name} not in known models. Using default hidden size 4096."
        )
        return 4096

    @staticmethod
    def _get_model_exponent(model_name: str) -> float:
        """Get the model-specific exponent P_m - matching tinker-cookbook."""
        model_lower = model_name.lower()

        if "llama" in model_lower:
            return HyperparamCalculator.MODEL_EXPONENTS["llama"]
        elif "qwen" in model_lower:
            return HyperparamCalculator.MODEL_EXPONENTS["qwen"]
        else:
            # tinker-cookbook raises an assertion error for unknown models
            logger.warning(
                f"Unknown model family for {model_name}. Using Llama exponent as fallback."
            )
            return HyperparamCalculator.MODEL_EXPONENTS["llama"]

    @staticmethod
    def get_recommended_lr(
        model_name: str, is_lora: bool = True, lora_rank: Optional[int] = None
    ) -> float:
        """
        Calculate optimal learning rate using tinker-cookbook's get_lr() implementation.

        Formula: LR(m) = lr_base · M_LoRA · (2000/H_m)^P_m

        This matches the exact implementation from:
        tinker_cookbook/hyperparam_utils.py::get_lr()

        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B")
            is_lora: Whether using LoRA (True) or full fine-tuning (False)
            lora_rank: LoRA rank (not used in formula, but kept for API compatibility)

        Returns:
            Recommended learning rate

        Note:
            From Tinker docs: "this function is independent of the LoRA rank"
        """
        base_lr = HyperparamCalculator.LR_BASE
        lora_multiplier = HyperparamCalculator.LORA_MULTIPLIER

        lr = base_lr * lora_multiplier if is_lora else base_lr
        hidden_size = HyperparamCalculator._get_hidden_size(model_name)
        exponent_model = HyperparamCalculator._get_model_exponent(model_name)

        lr = lr * ((2000 / hidden_size) ** exponent_model)

        logger.info(
            f"Calculated LR for {model_name}: {lr:.2e} "
            f"(hidden_size={hidden_size}, exponent={exponent_model}, is_lora={is_lora})"
        )

        return lr

    @staticmethod
    def get_recommended_batch_size(
        model_name: str, lora_rank: int = 32, recipe_type: str = "sft"
    ) -> int:
        """
        Recommend batch size based on model size and recipe.

        From Tinker docs:
        - Recommended size: 128 or smaller for supervised fine-tuning
        - Smaller batches yield better performance but require longer training
        - LoRA is less tolerant of high batch sizes than full fine-tuning

        Args:
            model_name: Model name
            lora_rank: LoRA rank
            recipe_type: Recipe type ("sft", "dpo", "rl")

        Returns:
            Recommended batch size
        """
        hidden_size = HyperparamCalculator._get_hidden_size(model_name)

        # Base recommendations from Tinker docs
        if recipe_type == "sft":
            # Supervised fine-tuning: 128 or smaller
            if hidden_size <= 2048:  # 1B models
                return 128
            elif hidden_size <= 4096:  # 8B models
                return 64
            else:  # 70B+ models
                return 32
        elif recipe_type == "dpo":
            # DPO typically uses smaller batches
            if hidden_size <= 2048:
                return 64
            elif hidden_size <= 4096:
                return 32
            else:
                return 16
        elif recipe_type == "rl":
            # RL uses much smaller batches
            if hidden_size <= 2048:
                return 32
            elif hidden_size <= 4096:
                return 16
            else:
                return 8
        else:
            # Default to SFT recommendations
            return 128

    @staticmethod
    def get_recommended_lora_rank(model_name: str, recipe_type: str = "sft") -> int:
        """
        Recommend LoRA rank based on use case.

        From Tinker docs:
        - Default: 32
        - RL: Small ranks (32) perform equivalently to larger ranks
        - SL on large datasets: Larger ranks where LoRA params >= completion tokens

        Args:
            model_name: Model name
            recipe_type: Recipe type

        Returns:
            Recommended LoRA rank
        """
        if recipe_type == "rl":
            return 32  # Small ranks work well for RL
        elif recipe_type == "sft":
            # For SFT, consider model size
            hidden_size = HyperparamCalculator._get_hidden_size(model_name)
            if hidden_size <= 2048:  # Small models
                return 32
            elif hidden_size <= 4096:  # Medium models
                return 64
            else:  # Large models
                return 128
        elif recipe_type == "dpo":
            return 32  # DPO typically uses smaller ranks
        else:
            return 32  # Safe default

    @staticmethod
    def get_all_recommendations(
        model_name: str,
        recipe_type: str = "sft",
        lora_rank: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Get all recommended hyperparameters at once.

        Args:
            model_name: HuggingFace model name
            recipe_type: Recipe type ("sft", "dpo", "rl")
            lora_rank: Override LoRA rank (if None, will be calculated)

        Returns:
            Dictionary of recommended hyperparameters with explanations
        """
        # Calculate or use provided LoRA rank
        if lora_rank is None:
            lora_rank = HyperparamCalculator.get_recommended_lora_rank(
                model_name, recipe_type
            )

        # Calculate learning rate (independent of LoRA rank)
        learning_rate = HyperparamCalculator.get_recommended_lr(
            model_name, is_lora=True, lora_rank=lora_rank
        )

        # Calculate batch size
        batch_size = HyperparamCalculator.get_recommended_batch_size(
            model_name, lora_rank, recipe_type
        )

        hidden_size = HyperparamCalculator._get_hidden_size(model_name)
        exponent = HyperparamCalculator._get_model_exponent(model_name)

        return {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "lora_rank": lora_rank,
            # Adam optimizer defaults (from tinker-cookbook)
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_eps": 1e-8,
            # Metadata for explanation
            "_metadata": {
                "hidden_size": hidden_size,
                "exponent": exponent,
                "lr_formula": f"LR for {model_name} = 5e-5 × 10 × (2000/{hidden_size})^{exponent:.4f} = {learning_rate:.2e}",
                "source": "https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams",
                "notes": [
                    "Learning rate is independent of LoRA rank",
                    "Batch size: 128 or smaller recommended for SFT",
                    "LoRA requires ~10x higher LR than full fine-tuning",
                ],
            },
        }
