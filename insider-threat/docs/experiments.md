# Experiment Plan

This document outlines reproducible experiments for improving insider threat detection models.

## Experiment Format

Each experiment follows this structure:
- **Goal**: What we're trying to achieve
- **Dataset**: Which data variant to use
- **Models**: Which models and hyperparameters to test
- **Metrics**: What to measure
- **Commands**: Exact commands to run
- **Results Location**: Where outputs are saved

## Baseline Experiments

### Exp-001: Baseline XGBoost

**Goal**: Establish baseline supervised model performance

**Dataset**: Full training set (data/features_train.csv)

**Models**:
- XGBoost with default hyperparameters

**Hyperparameters**:
- learning_rate: 0.1
- max_depth: 5
- n_estimators: 100

**Metrics**:
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

**Commands**:
```bash
python scripts/train_xgb.py \
  --input data/features_train.csv \
  --test_path data/features_test.csv \
  --n_estimators 100 \
  --learning_rate 0.1 \
  --max_depth 5

python scripts/evaluate.py --test_data data/features_test.csv
```

**Results Location**: `artifacts/exp-001/`

---

### Exp-002: Baseline Isolation Forest

**Goal**: Establish baseline unsupervised model performance

**Dataset**: Full training set (data/features_train.csv)

**Models**:
- Isolation Forest with contamination=0.01

**Hyperparameters**:
- contamination: 0.01 (1% expected anomalies)
- n_estimators: 100

**Metrics**:
- Precision@1%, Recall@1%
- Precision@5%, Recall@5%
- Top-k precision

**Commands**:
```bash
python scripts/train_iso.py \
  --input data/features_train.csv \
  --contamination 0.01 \
  --n_estimators 100

python scripts/evaluate.py --test_data data/features_test.csv
```

**Results Location**: `artifacts/exp-002/`

---

## Hyperparameter Tuning Experiments

### Exp-003: XGBoost Hyperparameter Grid Search

**Goal**: Find optimal XGBoost hyperparameters

**Dataset**: Full training set

**Hyperparameter Grid**:
```python
learning_rate: [0.01, 0.05, 0.1]
max_depth: [3, 5, 7]
n_estimators: [100, 500]
```

**Total Combinations**: 3 × 3 × 2 = 18

**Metrics**:
- ROC-AUC (primary)
- F1-Score
- Training time

**Commands**:
```bash
python scripts/hparam_search.py \
  --model xgb \
  --input data/features_train.csv \
  --test_path data/features_test.csv \
  --output artifacts/exp-003/hparam_results.csv
```

**Results Location**: `artifacts/exp-003/`

---

### Exp-004: Isolation Forest Contamination Tuning

**Goal**: Find optimal contamination rate

**Dataset**: Full training set

**Hyperparameters**:
- contamination: [0.001, 0.005, 0.01, 0.02, 0.05]

**Metrics**:
- Precision@1%
- Recall@1%
- F1-Score@1%

**Commands**:
```bash
for contam in 0.001 0.005 0.01 0.02 0.05; do
  python scripts/train_iso.py \
    --input data/features_train.csv \
    --contamination $contam \
    --output_model models/iso_model_contam_${contam}.pkl
done
```

**Results Location**: `artifacts/exp-004/`

---

## Feature Engineering Experiments

### Exp-005: Additional Features

**Goal**: Test impact of additional behavioral features

**New Features to Add**:
- n-gram of file paths (common file access patterns)
- Time between events (burst detection)
- Weekend/weekday indicator
- Hour of day statistics (mean, std)

**Commands**:
```bash
# Modify data_prep.py to add new features
python scripts/data_prep.py --input data/cert_dataset.csv --output data/features_extended.csv

python scripts/train_xgb.py \
  --input data/features_extended_train.csv \
  --test_path data/features_extended_test.csv
```

**Results Location**: `artifacts/exp-005/`

---

## Model Comparison Experiments

### Exp-006: Ensemble Methods

**Goal**: Combine XGBoost and Isolation Forest predictions

**Approach**:
- Average probability scores
- Weighted voting
- Stacking

**Commands**:
```bash
# Train both models
python scripts/train_xgb.py ...
python scripts/train_iso.py ...

# Combine predictions
python scripts/ensemble.py \
  --xgb_model models/xgb_model.pkl \
  --iso_model models/iso_model.pkl \
  --test_data data/features_test.csv
```

**Results Location**: `artifacts/exp-006/`

---

## Hyperparameter Search Script

Create `scripts/hparam_search.py`:

```python
"""
Hyperparameter Grid Search Script

Purpose: Systematically test hyperparameter combinations and find optimal settings.

Usage:
    python scripts/hparam_search.py --model xgb --input data/features_train.csv
"""

import pandas as pd
import itertools
import json
from pathlib import Path
# ... (implementation would go here)
```

## Analyzing Results

### How to Pick Next Experiment

1. **Compare ROC-AUC**: Higher is better for overall performance
2. **Check Precision@1%**: For anomaly detection, high precision means fewer false alarms
3. **Look at Training Time**: Balance performance vs. speed
4. **Review Feature Importance**: If certain features dominate, consider feature engineering

### Example Analysis Workflow

```bash
# Compare all experiments
python scripts/compare_experiments.py \
  --results_dir artifacts/ \
  --output artifacts/experiment_summary.csv

# Visualize best hyperparameters
python scripts/plot_hparam_results.py \
  --input artifacts/exp-003/hparam_results.csv
```

## Experiment Tracking

### Recommended Structure

```
artifacts/
├── exp-001/
│   ├── metrics.json
│   ├── model.pkl
│   └── plots/
├── exp-002/
│   └── ...
└── experiment_summary.csv
```

### Logging Best Practices

- Save model configs with each experiment
- Record training time and resource usage
- Document any data preprocessing steps
- Note any anomalies or issues encountered

## Next Steps After Experiments

1. **Select Best Model**: Based on precision@1% and ROC-AUC
2. **Production Deployment**: Use best model for API
3. **Monitor Performance**: Track model performance on real data
4. **Iterate**: Use production feedback to design next experiments

