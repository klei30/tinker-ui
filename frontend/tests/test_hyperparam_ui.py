"""
Integration tests for hyperparameter calculation from UI perspective.
"""

import pytest
from playwright.sync_api import Page, expect


class TestHyperparamUIIntegration:
    """Test hyperparameter calculation from the UI perspective."""

    def test_hyperparam_calculator_ui_loads(self, page: Page):
        """Test that the hyperparameter calculator UI loads properly."""
        # Navigate to the main page
        page.goto("http://localhost:3000")

        # Check that the page loads
        expect(page).to_have_title("Tinker Platform Dashboard")

        # The hyperparameter calculator should be accessible through the run creation wizard
        # This would require setting up the full UI flow

    def test_hyperparam_calculation_workflow(self, page: Page):
        """Test the complete hyperparameter calculation workflow from UI."""
        # This would test:
        # 1. Opening the run creation wizard
        # 2. Selecting a model
        # 3. Choosing a recipe type
        # 4. Seeing hyperparameter recommendations
        # 5. Verifying the API call was made correctly

        # For now, we'll create a basic structure that can be expanded
        page.goto("http://localhost:3000")

        # Wait for the page to be fully loaded
        expect(page.locator("text=Tinker Platform")).to_be_visible()

        # This test would need to be expanded with actual UI interactions
        # once the hyperparameter calculator component is implemented in the UI

    def test_hyperparam_api_integration(self, page: Page):
        """Test that the UI correctly calls the hyperparameter API."""
        # Mock the API response
        page.route(
            "**/hyperparameters/calculate",
            lambda route: route.fulfill(
                json={
                    "success": True,
                    "model_name": "meta-llama/Llama-3.1-8B",
                    "recipe_type": "sft",
                    "recommendations": {
                        "learning_rate": 2.86e-04,
                        "batch_size": 64,
                        "lora_rank": 64,
                    },
                    "explanation": {
                        "learning_rate": "LR = 5e-5 × 10 × (2000/4096)^0.781 = 2.86e-04",
                        "batch_size": "Optimized for sft training. Tinker docs recommend 128 or smaller for SFT.",
                        "lora_rank": "Default is 32 for most use cases. Independent of learning rate.",
                        "notes": ["Learning rate is independent of LoRA rank"],
                        "source": "https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams",
                    },
                }
            ),
        )

        # Navigate to page
        page.goto("http://localhost:3000")

        # This test would need to be expanded to actually trigger the API call
        # through UI interactions

    def test_hyperparam_error_handling_ui(self, page: Page):
        """Test that the UI properly handles hyperparameter calculation errors."""
        # Mock an API error response
        page.route(
            "**/hyperparameters/calculate",
            lambda route: route.fulfill(
                json={"detail": "Failed to calculate hyperparameters: Test error"},
                status=500,
            ),
        )

        page.goto("http://localhost:3000")

        # This test would verify that errors are displayed properly in the UI
        # when hyperparameter calculation fails
