# Continuous Integration Guide

This document explains how the CI/CD workflow works and how to add new checks.

## Current CI Pipeline

The CI pipeline (`.github/workflows/ci.yml`) runs automatically on:
- Every push to `main` or `develop` branches
- Every pull request targeting `main` or `develop`

### Steps

1. **Checkout Code**: Gets the latest code from the repository
2. **Set up Python**: Installs Python 3.10 and caches pip packages
3. **Install Dependencies**: Installs packages from `requirements.txt` plus pytest and flake8
4. **Run Tests**: Executes all tests in `tests/` directory
5. **Lint Code**: Runs flake8 to check code style and catch errors

## Adding New Checks

### Example: Add Coverage Report

```yaml
- name: Generate coverage report
  run: |
    pip install pytest-cov
    pytest --cov=scripts --cov=app --cov-report=xml tests/
  
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Example: Add Type Checking

```yaml
- name: Type check with mypy
  run: |
    pip install mypy
    mypy scripts/ app/
```

### Example: Add Security Scanning

```yaml
- name: Run security scan
  uses: securego/gosec@master
  # Or use bandit for Python
  run: |
    pip install bandit
    bandit -r scripts/ app/
```

## Running CI Locally

You can test CI checks locally before pushing:

```bash
# Install CI dependencies
pip install pytest flake8

# Run tests
pytest tests/ -v

# Run linter
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Troubleshooting CI Failures

### Tests Failing

1. Check test output in GitHub Actions logs
2. Run tests locally: `pytest tests/ -v`
3. Fix failing tests before pushing

### Linting Errors

1. Check flake8 output for specific errors
2. Fix formatting issues
3. Consider using `black` or `autopep8` for auto-formatting

### Dependency Issues

1. Ensure `requirements.txt` includes all test dependencies
2. Check Python version compatibility
3. Update `python-version` in workflow if needed

## Best Practices

1. **Run tests locally first**: Don't rely on CI to catch errors
2. **Fix linting issues**: Keep code style consistent
3. **Keep CI fast**: Add only necessary checks to avoid slow builds
4. **Cache dependencies**: Use pip caching (already configured)
5. **Fail fast**: Run quick checks first, slow checks later

