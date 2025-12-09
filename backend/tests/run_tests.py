"""
Run hyperparameter tests.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all hyperparameter tests."""
    test_dir = Path(__file__).parent / "tests"

    # Run unit tests
    print("Running unit tests...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_dir / "test_hyperparam_calculator.py"),
            "-v",
        ],
        capture_output=True,
        text=True,
    )

    print("Unit test output:")
    print(result.stdout)
    if result.stderr:
        print("Unit test errors:")
        print(result.stderr)

    # Run API tests
    print("\nRunning API tests...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_dir / "test_hyperparam_api.py"),
            "-v",
        ],
        capture_output=True,
        text=True,
    )

    print("API test output:")
    print(result.stdout)
    if result.stderr:
        print("API test errors:")
        print(result.stderr)

    # Run E2E tests (if backend is running)
    print("\nRunning E2E tests...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_dir / "test_hyperparam_e2e.py"),
            "-v",
            "-k",
            "not e2e",  # Skip actual E2E tests for now
        ],
        capture_output=True,
        text=True,
    )

    print("E2E test output:")
    print(result.stdout)
    if result.stderr:
        print("E2E test errors:")
        print(result.stderr)


if __name__ == "__main__":
    run_tests()
