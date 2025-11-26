# Tinker UI Backend Test Suite

Comprehensive test suite for all features of the Tinker UI backend application.

## Overview

This test suite provides complete coverage of:
- **API Endpoints** - All FastAPI REST endpoints
- **Training Workflows** - End-to-end training for all recipe types
- **Evaluation** - Model evaluation functionality
- **Utilities** - Text, environment, and recipe executor utilities
- **Dataset Processing** - Format detection and conversion
- **Checkpoint Management** - Checkpoint creation and storage
- **Training Types** - All supported training recipes

## Test Files

### 1. `conftest.py` - Shared Fixtures
Comprehensive fixtures for all tests including:
- Database session mocks
- Model fixtures (User, Project, Dataset, Run, Checkpoint, Evaluation)
- Job runner fixtures
- API client fixtures
- Mock API fixtures (Tinker API, renderer, tokenizer)
- Dataset format fixtures
- Environment fixtures

### 2. `test_api_endpoints.py` - API Endpoint Tests
**31 tests** covering all FastAPI endpoints:
- Health & Status (2 tests)
- Project Management (3 tests)
- Dataset Management (5 tests)
- Run Management (6 tests)
- Checkpoint Endpoints (1 test)
- Hyperparameter Endpoints (2 tests)
- Model Management (3 tests)
- Chat & Sampling (2 tests)
- Evaluation (2 tests)
- Metrics & Logs (4 tests)
- Cost Estimation (1 test)

### 3. `test_training_workflows.py` - Training Workflow Tests
**19 integration tests** for end-to-end training workflows:
- SFT Workflow (3 tests)
- DPO Workflow (2 tests)
- RL Workflow (2 tests)
- CHAT_SL Workflow (1 test)
- DISTILLATION Workflow (1 test)
- MATH_RL Workflow (1 test)
- ON_POLICY_DISTILLATION Workflow (1 test)
- Multi-Recipe Workflow (2 tests)
- Error Recovery (2 tests)
- Resume from Checkpoint (2 tests)
- Progress Tracking (2 tests)

### 4. `test_evaluation.py` - Evaluation Tests
**27 tests** for model evaluation:
- Evaluation Creation (3 tests)
- Renderer Integration (3 tests)
- Response Parsing (3 tests)
- Metrics Collection (3 tests)
- Test Prompt Handling (3 tests)
- Error Handling (4 tests)
- Multi-Model Evaluation (2 tests)
- Benchmark Evaluations (3 tests)
- Evaluation Persistence (3 tests)

### 5. `test_utils.py` - Utility Module Tests
**38 tests** for utility modules:
- Text Utilities (11 tests)
  - Strip ANSI codes
  - Truncate text
  - Sanitize filenames
- Environment Utilities (11 tests)
  - Get required/optional env vars
  - Setup training environment
  - Validate API keys
- Recipe Executor (11 tests)
  - Create executor
  - Log writing
  - Parse metrics
  - Execute recipes
- Error Handling (3 tests)
- Integration (2 tests)

### 6. `test_dataset_processing.py` - Dataset Tests
**29 tests** for dataset format handling:
- Format Detection (3 tests)
- Alpaca Format Processing (3 tests)
- Messages Format Processing (3 tests)
- Mixed Format Handling (2 tests)
- Dataset Validation (4 tests)
- Data Conversion (3 tests)
- Error Handling (6 tests)
- Dataset Filtering (3 tests)
- Dataset Preview (2 tests)

### 7. `test_checkpoint_management.py` - Checkpoint Tests
**27 tests** for checkpoint management:
- Checkpoint Creation (3 tests)
- Checkpoint Registration (3 tests)
- Checkpoint Storage (3 tests)
- Checkpoint Retrieval (3 tests)
- Checkpoint Metrics (3 tests)
- Checkpoint Selection (3 tests)
- Checkpoint Download (3 tests)
- Error Handling (5 tests)
- Checkpoint Lifecycle (1 test)

### 8. `test_training_types.py` - Training Types Tests
**20 tests** (existing) for training type validation:
- Training Type Differentiation (5 tests)
- Dataset Format Handling (3 tests)
- API Key Validation (2 tests)
- Error Handling (3 tests)
- Race Condition Prevention (2 tests)
- Stub Implementation Detection (1 test)
- Integration Tests (2 tests)
- Performance Tests (2 tests)

## Total Test Coverage

| Category | Test Files | Test Count |
|----------|-----------|------------|
| API Endpoints | 1 | 31 tests |
| Training Workflows | 1 | 19 tests |
| Evaluation | 1 | 27 tests |
| Utilities | 1 | 38 tests |
| Dataset Processing | 1 | 29 tests |
| Checkpoint Management | 1 | 27 tests |
| Training Types | 1 | 20 tests |
| **TOTAL** | **8 files** | **191 tests** |

## Running Tests

### Run All Tests
```bash
cd backend
pytest tests/ -v
```

### Run Specific Test File
```bash
# API endpoint tests
pytest tests/test_api_endpoints.py -v

# Training workflow tests
pytest tests/test_training_workflows.py -v

# Evaluation tests
pytest tests/test_evaluation.py -v

# Utility tests
pytest tests/test_utils.py -v

# Dataset processing tests
pytest tests/test_dataset_processing.py -v

# Checkpoint management tests
pytest tests/test_checkpoint_management.py -v

# Training types tests
pytest tests/test_training_types.py -v
```

### Run by Test Category
```bash
# Unit tests only (fast)
pytest tests/ -v -m "not integration and not e2e"

# Integration tests only
pytest tests/ -v -m integration

# End-to-end tests only
pytest tests/ -v -m e2e

# Exclude slow tests
pytest tests/ -v -m "not slow"
```

### Run Specific Test Class
```bash
pytest tests/test_api_endpoints.py::TestRunEndpoints -v
pytest tests/test_training_workflows.py::TestSFTWorkflow -v
pytest tests/test_evaluation.py::TestRendererIntegration -v
```

### Run Specific Test Function
```bash
pytest tests/test_api_endpoints.py::TestRunEndpoints::test_create_run_success -v
```

## Coverage Reports

### Generate Coverage Report
```bash
# Run tests with coverage
pytest tests/ --cov=. --cov-report=html --cov-report=term

# View HTML report
# Open htmlcov/index.html in browser
```

### Coverage by Module
```bash
# Coverage for specific module
pytest tests/ --cov=job_runner --cov-report=term
pytest tests/ --cov=main --cov-report=term
pytest tests/ --cov=utils --cov-report=term
```

## Test Markers

Tests are marked for categorization:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Integration tests (may require external services)
- `@pytest.mark.e2e` - End-to-end tests (full workflows)
- `@pytest.mark.slow` - Slow tests (>1 second)
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.asyncio` - Async tests
- `@pytest.mark.requires_api_key` - Tests requiring TINKER_API_KEY
- `@pytest.mark.requires_gpu` - Tests requiring GPU access

## Environment Setup

### Required Environment Variables
```bash
# For tests that require API key
export TINKER_API_KEY="tml-your-api-key-here"

# For testing mode
export TESTING="true"

# Python IO encoding
export PYTHONIOENCODING="utf-8"
```

### Optional: Setup Test Environment
```python
from utils.env_utils import setup_test_environment

setup_test_environment()
```

## Test Dependencies

All test dependencies are included in the main requirements. Key testing packages:

- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities
- `httpx` - Async HTTP client for API tests

## Continuous Integration

### GitHub Actions / CI Pipeline
```yaml
# Example CI configuration
- name: Run tests
  run: |
    cd backend
    pytest tests/ -v --cov=. --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Writing New Tests

### Test Structure
```python
import pytest
from unittest.mock import Mock, patch, AsyncMock

class TestNewFeature:
    """Test description."""

    def test_feature_success(self, fixture1, fixture2):
        """Test successful case."""
        # Arrange
        ...

        # Act
        result = function_under_test()

        # Assert
        assert result == expected

    @pytest.mark.asyncio
    async def test_feature_async(self):
        """Test async functionality."""
        result = await async_function()
        assert result is not None
```

### Use Existing Fixtures
```python
def test_with_fixtures(
    mock_session,
    sample_run,
    sample_dataset,
    mock_env_vars
):
    """Test using shared fixtures."""
    # Fixtures are automatically available
    assert sample_run.id == 1
```

## Troubleshooting

### Tests Fail Due to Missing Modules
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov
```

### Tests Fail Due to Import Errors
```python
# Add backend to path in test file
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

### Async Tests Fail
```python
# Ensure test is marked as async
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result is not None
```

### Mock Not Working
```python
# Use patch correctly
with patch("module.function") as mock_func:
    mock_func.return_value = "test"
    result = function_that_calls_it()
```

## Best Practices

1. **Isolation** - Each test should be independent
2. **Fixtures** - Use shared fixtures from conftest.py
3. **Mocking** - Mock external dependencies (API calls, file I/O)
4. **Naming** - Use descriptive test names (test_<what>_<when>_<expected>)
5. **AAA Pattern** - Arrange, Act, Assert
6. **Fast Tests** - Keep unit tests fast (<100ms)
7. **Coverage** - Aim for >80% code coverage
8. **Documentation** - Document complex test scenarios

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure all tests pass
3. Maintain >80% coverage
4. Add test documentation
5. Update this README if adding new test files

## Questions?

For questions about tests or contributing, please see:
- Main project README
- Code contribution guidelines
- Issue tracker
