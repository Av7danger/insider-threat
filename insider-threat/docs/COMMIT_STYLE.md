# Commit Message Style Guide

This project uses conventional commits for consistency and automatic changelog generation.

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Scope (Optional)

Area of codebase affected:
- `scripts`: Training or utility scripts
- `app`: API code
- `tests`: Test files
- `docs`: Documentation
- `docker`: Docker configuration

## Subject

- Use imperative mood: "Add feature" not "Added feature"
- Keep under 50 characters
- No period at end
- Capitalize first letter

## Body (Optional)

- Explain what and why, not how
- Wrap at 72 characters
- Reference issues: "Fixes #123"

## Examples

### Good Commits

```
feat(scripts): Add LSTM sequence model training

Implements PyTorch LSTM for detecting temporal patterns in user behavior.
Supports GPU training with automatic device detection.

Closes #45
```

```
fix(app): Handle missing models gracefully in API

Returns 503 error instead of crashing when models are not found.
Improves error messages for debugging.

Fixes #67
```

```
docs(readme): Update installation instructions

Adds Python 3.10 requirement and virtual environment setup steps.
```

```
test(scripts): Add unit tests for data_prep.py

Covers feature engineering and missing value handling.
```

### Bad Commits

```
❌ "fixed bug" - Too vague
❌ "WIP" - Not descriptive
❌ "Update files" - Doesn't explain what changed
❌ "Fixed the thing" - Unclear what "thing" is
```

## Simple Alternative

If conventional commits feel too formal, use this simpler format:

```
<type>: <description>
```

Examples:
- `feat: Add XGBoost training script`
- `fix: Handle missing timestamp column`
- `docs: Update README with Docker instructions`

## Initial Project Commits

For the initial project setup, example commits:

1. `feat: Add project scaffold and directory structure`
2. `feat(scripts): Add schema detection and data preparation scripts`
3. `feat(scripts): Add Isolation Forest and XGBoost training scripts`
4. `feat(app): Add FastAPI inference service`
5. `feat(docker): Add Dockerfile and docker-compose configuration`
6. `test: Add unit tests for core functionality`
7. `docs: Add comprehensive README and tutorials`

## Tips

- Make atomic commits (one logical change per commit)
- Write clear, descriptive commit messages
- Review your commits before pushing
- Use `git commit --amend` to fix the last commit message

