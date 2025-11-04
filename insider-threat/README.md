# Insider Threat Detection using Machine Learning

This project implements a comprehensive ML-based insider threat detection system using multiple approaches: unsupervised anomaly detection (Isolation Forest), supervised classification (XGBoost), and sequence modeling (LSTM). The system analyzes user activity patterns to identify potentially malicious insider behavior.

## Quick Start (For Beginners)

### 1. Setup Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your CERT dataset CSV file at `data/cert_dataset.csv`. If you don't have the dataset, a small synthetic sample is provided at `data/sample_cert_small.csv` for testing.

Verify the dataset:
```bash
# On Windows PowerShell:
bash scripts/verify_dataset.sh
# Or on Linux/Mac:
chmod +x scripts/verify_dataset.sh
./scripts/verify_dataset.sh
```

### 3. Detect Schema and Inventory

```bash
python scripts/schema_and_inventory.py data/cert_dataset.csv
```

**Expected Output:** `data/detected_schema.json` with column information and type inference.

### 4. Prepare Features

```bash
python scripts/data_prep.py --input data/cert_dataset.csv --output data/features.csv
```

**Expected Output:** 
- `data/features.csv` - aggregated features per user-day
- `data/features_train.csv` - training split (80%)
- `data/features_test.csv` - test split (20%)

### 5. Train Models

**Train Isolation Forest (Unsupervised):**
```bash
python scripts/train_iso.py --input data/features_train.csv --contamination 0.01
```

**Expected Output:** 
- `models/iso_model.pkl` - trained Isolation Forest model
- `models/iso_scaler.pkl` - feature scaler
- `models/iso_train_scores.csv` - anomaly scores for training data

**Train XGBoost (Supervised, if labels exist):**
```bash
python scripts/train_xgb.py --input data/features_train.csv --test_path data/features_test.csv
```

**Expected Output:**
- `models/xgb_model.pkl` - trained XGBoost classifier
- `models/xgb_scaler.pkl` - feature scaler
- `artifacts/xgb_metrics.json` - evaluation metrics
- `artifacts/xgb_confusion_matrix.png` - confusion matrix plot

**Train LSTM (Sequence Model, if labels exist):**
```bash
python scripts/train_lstm.py --input data/cert_dataset.csv --epochs 10
```

**Expected Output:**
- `models/lstm_model.pt` - trained PyTorch LSTM model
- `artifacts/lstm_training_curve.png` - training loss curve

### 6. Evaluate Models

```bash
python scripts/evaluate.py --test_data data/features_test.csv
```

**Expected Output:**
- `artifacts/model_comparison.md` - detailed comparison report
- `artifacts/summary_metrics.csv` - metrics summary table
- `artifacts/roc.png` - ROC curves for all models

### 7. Generate Explanations (SHAP)

```bash
python scripts/explain_xgb_shap.py --model_path models/xgb_model.pkl --test_data data/features_test.csv
```

**Expected Output:**
- `artifacts/shap_summary.png` - feature importance summary plot
- `artifacts/shap_force_*.html` - individual prediction explanations

### 8. Start Inference API

```bash
# On Windows PowerShell:
bash scripts/run_api.sh
# Or on Linux/Mac:
chmod +x scripts/run_api.sh
./scripts/run_api.sh
```

**Expected Output:** API server running on `http://localhost:8000`

Test the API:
```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_prediction_request.json
```

### 9. Run Tests

```bash
pytest tests/
```

## Docker Deployment

### Build and Run with Docker

```bash
cd docker
docker-compose up --build
```

See `docker/README.md` for detailed instructions.

## Project Structure

```
insider-threat/
├── data/                 # Dataset files (CSV)
├── models/               # Saved model files (.pkl, .pt)
├── notebooks/            # Jupyter notebooks for exploration
├── scripts/              # Training and utility scripts
├── app/                  # FastAPI application
├── docker/               # Docker configuration
├── tests/                # Unit tests
├── docs/                 # Documentation
└── artifacts/            # Evaluation results, plots, metrics
```

## Learning Path

By running through this project, beginners will learn:

1. **Data Engineering**: Schema detection, feature engineering, time-series aggregation
2. **Unsupervised Learning**: Isolation Forest for anomaly detection without labels
3. **Supervised Learning**: XGBoost for classification with labeled data
4. **Deep Learning**: LSTM for sequence modeling of user behavior
5. **Model Evaluation**: Metrics for imbalanced classification, ROC curves, precision@k
6. **Explainability**: SHAP values for understanding model decisions
7. **MLOps**: FastAPI deployment, Docker containerization, CI/CD workflows
8. **Best Practices**: Testing, documentation, experiment tracking

## Next Steps

- See `docs/tutorial_for_beginners.md` for a step-by-step walkthrough
- Check `docs/experiments.md` for experiment ideas and hyperparameter tuning
- Review `docs/TASKS.md` for follow-up tasks and improvements

## Troubleshooting

Common issues and solutions are documented in:
- `docs/tutorial_for_beginners.md` (Common Gotchas section)
- `docker/README.md` (Docker troubleshooting)

## Contributing

See `.github/PULL_REQUEST_TEMPLATE.md` for contribution guidelines.

