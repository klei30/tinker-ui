"""
Frontend test runner script.
"""

import subprocess
import sys
from pathlib import Path


def run_frontend_tests():
    """Run frontend hyperparameter tests."""
    test_dir = Path(__file__).parent / "tests"

    print("Running frontend hyperparameter tests...")

    # Install test dependencies if needed
    print("Installing test dependencies...")
    result = subprocess.run(
        ["pnpm", "install"], cwd=test_dir, capture_output=True, text=True
    )

    if result.returncode != 0:
        print("Failed to install test dependencies:")
        print(result.stderr)
        return False

    # Run the tests
    print("Running tests...")
    result = subprocess.run(
        ["pnpm", "test:run"], cwd=test_dir, capture_output=True, text=True
    )

    print("Test output:")
    print(result.stdout)

    if result.stderr:
        print("Test errors:")
        print(result.stderr)

    return result.returncode == 0


if __name__ == "__main__":
    success = run_frontend_tests()
    sys.exit(0 if success else 1)
