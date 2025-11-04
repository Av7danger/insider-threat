# Release Checklist

Use this checklist before merging to main or creating a release.

## Pre-Merge Validation

### Code Quality
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code passes linting: `flake8 .`
- [ ] No syntax errors or warnings
- [ ] Code is properly commented and documented

### Models
- [ ] Trained models saved to `models/` directory
- [ ] Model files have descriptive names (e.g., `xgb_model.pkl`, `iso_model.pkl`)
- [ ] Scalers are saved alongside models
- [ ] Model versions are documented (if applicable)

### Data
- [ ] Example data included OR clear instructions to fetch dataset
- [ ] Sample synthetic data available for testing (`data/sample_cert_small.csv`)
- [ ] Data paths are documented in README

### Artifacts
- [ ] Evaluation metrics saved to `artifacts/`
- [ ] Plots and visualizations generated
- [ ] Model comparison report exists
- [ ] SHAP explanations generated (if applicable)

### Documentation
- [ ] README.md updated with latest commands
- [ ] All scripts have docstrings and usage examples
- [ ] API documentation is accurate
- [ ] Tutorial for beginners is complete

### Testing
- [ ] Unit tests cover core functionality
- [ ] Integration tests pass (if applicable)
- [ ] API endpoints tested manually or with automated tests
- [ ] Docker build and run successfully

### CI/CD
- [ ] GitHub Actions workflow passes
- [ ] No failing checks in pull request
- [ ] Release notes prepared (if creating a release)

## Sample Data Check

Before releasing, ensure users can test the system:

- [ ] Synthetic sample dataset is available
- [ ] Sample dataset is small enough to download quickly
- [ ] Sample dataset includes both normal and anomalous examples (if applicable)
- [ ] README explains how to use sample data

## Release Notes Template

When creating a release, include:

1. **New Features**: What's new in this release
2. **Improvements**: Enhancements to existing features
3. **Bug Fixes**: Issues resolved
4. **Breaking Changes**: Any changes that require user action
5. **Model Performance**: Key metrics (if models are updated)

## Post-Release

After merging to main:

- [ ] Tag release: `git tag -a v1.0.0 -m "Release v1.0.0"`
- [ ] Push tags: `git push origin v1.0.0`
- [ ] Update CHANGELOG.md (if maintained)
- [ ] Announce release (if applicable)

