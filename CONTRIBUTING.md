# Contributing to Tinker UI

Thank you for your interest in contributing to Tinker UI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

This project follows a standard Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Accept responsibility and apologize for mistakes
- Focus on what is best for the community

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tinker-ui.git
   cd tinker-ui
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/tinker-ui.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

### Frontend Setup

```bash
cd frontend
pnpm install
```

### Environment Configuration

Create the necessary `.env` files as described in the main [README.md](README.md).

## Development Workflow

1. **Keep your fork up to date**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following our coding standards

4. **Test your changes**:
   ```bash
   # Backend tests
   cd backend
   pytest

   # Frontend tests
   cd frontend/tests
   pnpm test:full
   ```

5. **Commit your changes** with clear commit messages

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Coding Standards

### Python (Backend)

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use type hints where possible
- Maximum line length: 100 characters
- Use `ruff` for linting and formatting:
  ```bash
  ruff check .
  ruff format .
  ```
- Write docstrings for all public functions and classes
- Use meaningful variable and function names

### TypeScript/JavaScript (Frontend)

- Follow the existing code style
- Use TypeScript for type safety
- Use functional components and hooks
- Maximum line length: 100 characters
- Use ESLint and Prettier:
  ```bash
  pnpm lint
  pnpm format
  ```
- Prefer named exports over default exports
- Use meaningful variable and function names

### General Guidelines

- Keep functions small and focused (single responsibility)
- Avoid code duplication (DRY principle)
- Write self-documenting code with clear names
- Add comments only when necessary to explain "why" not "what"
- Handle errors appropriately
- Validate inputs at system boundaries

## Testing Guidelines

### Backend Testing

- Write tests for all new features and bug fixes
- Follow the existing test structure in `backend/tests/`
- Use pytest fixtures for setup and teardown
- Use markers to categorize tests:
  - `@pytest.mark.unit` - Fast, isolated unit tests
  - `@pytest.mark.integration` - Integration tests
  - `@pytest.mark.e2e` - End-to-end tests
  - `@pytest.mark.slow` - Tests that take >1 second

Example test:
```python
import pytest

@pytest.mark.unit
def test_example_function(mock_session):
    """Test that example_function returns expected result."""
    result = example_function(mock_session, "input")
    assert result == "expected_output"
```

### Frontend Testing

- Write tests for UI components
- Use Vitest and React Testing Library
- Test user interactions, not implementation details
- Aim for meaningful test coverage

Example test:
```typescript
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

describe('Component', () => {
  it('renders correctly', () => {
    render(<Component />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });
});
```

### Test Coverage

- Maintain or improve overall test coverage
- Focus on critical paths and edge cases
- Run coverage reports:
  ```bash
  pytest --cov=. --cov-report=html
  ```

## Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Examples

```
feat(api): add endpoint for hyperparameter calculation

Implement new API endpoint to calculate optimal hyperparameters
based on model size and training recipe type.

Closes #123
```

```
fix(ui): resolve progress bar not updating during training

The progress bar was stuck at 0% because the backend wasn't
properly polling the metrics file. Added fallback calculation
based on step count.

Fixes #456
```

```
docs(readme): update testing instructions

Add detailed instructions for running backend and frontend tests,
including coverage reports and test categories.
```

### Commit Message Tips

- Use imperative mood ("add" not "added" or "adds")
- First line should be 50 characters or less
- Separate subject from body with a blank line
- Wrap body at 72 characters
- Reference issues and pull requests in the footer

## Pull Request Process

1. **Update documentation** if you've made changes to APIs or features

2. **Add tests** for new functionality or bug fixes

3. **Run the test suite** and ensure all tests pass:
   ```bash
   # Backend
   cd backend && pytest

   # Frontend
   cd frontend/tests && pnpm test:full
   ```

4. **Update the README** if you've added new features or changed setup

5. **Fill out the PR template** with all required information

6. **Request review** from maintainers

7. **Address feedback** and make requested changes

8. **Ensure CI passes** (if GitHub Actions are set up)

### PR Requirements

- [ ] Code follows project coding standards
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow guidelines
- [ ] No merge conflicts with main branch
- [ ] Code has been reviewed for security issues
- [ ] Breaking changes are clearly documented

### PR Title Format

Use the same format as commit messages:
```
feat(component): brief description of changes
```

## Reporting Bugs

### Before Submitting

1. **Check existing issues** to avoid duplicates
2. **Test with the latest version** to see if the bug still exists
3. **Collect information**:
   - Operating system and version
   - Python version
   - Node.js version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and stack traces
   - Screenshots if applicable

### Bug Report Template

```markdown
## Description
A clear description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Windows 10, macOS 13, Ubuntu 22.04]
- Python: [e.g., 3.11.5]
- Node.js: [e.g., 18.17.0]
- Browser: [e.g., Chrome 118] (if applicable)

## Additional Context
Any other relevant information, screenshots, or logs.
```

## Suggesting Features

### Before Suggesting

1. **Check existing issues** to see if someone already suggested it
2. **Consider if it fits** the project's scope and goals
3. **Think about implementation** - how would this work?

### Feature Request Template

```markdown
## Feature Description
A clear description of the feature you'd like to see.

## Use Case
Explain why this feature would be useful. Who would benefit?

## Proposed Solution
Describe how you envision this working.

## Alternatives Considered
What other solutions did you consider?

## Additional Context
Any mockups, examples, or related information.
```

## Development Tips

### Backend Development

- Use virtual environments to isolate dependencies
- Run the development server with auto-reload:
  ```bash
  uvicorn main:app --reload --log-level debug
  ```
- Use logging instead of print statements
- Test API endpoints with tools like Postman or curl

### Frontend Development

- Use the Next.js development server:
  ```bash
  pnpm dev
  ```
- Install the React DevTools browser extension
- Check the browser console for errors
- Use TypeScript to catch errors early

### Database Changes

- When modifying models, consider migration impact
- Test database changes thoroughly
- Document any schema changes

## Questions?

If you have questions or need help:

1. Check the [README.md](README.md) and documentation
2. Search existing issues and discussions
3. Create a new discussion or issue
4. Reach out to maintainers

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes (for significant contributions)
- Project README (for major features)

Thank you for contributing to Tinker UI!
