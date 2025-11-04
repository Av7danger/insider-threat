# Quick Start Guide - Exact Commands

This document provides the **exact shell commands** a beginner should run to get the entire project working from scratch.

## Prerequisites

- Python 3.10 or higher installed
- Git (optional, for version control)

## Step-by-Step Commands

### 1. Setup Environment

```bash
# Navigate to project directory
cd insider-threat

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Expected Output**: You should see packages being installed. When done, `(venv)` appears in your prompt.

---

### 2. Verify Dataset

```bash
# On Windows PowerShell:
bash scripts/verify_dataset.sh

# On Linux/Mac:
chmod +x scripts/verify_dataset.sh
./scripts/verify_dataset.sh
```

**Expected Output**: 
```
✓ Dataset file found: data/cert_dataset.csv
Rows: 15
Columns: 7
```

**Note**: If you don't have the full dataset, the script will use `data/sample_cert_small.csv` for testing.

---

### 3. Detect Schema

```bash
python scripts/schema_and_inventory.py data/cert_dataset.csv
```

**Expected Output**: Schema summary with column types and sample values. Creates `data/detected_schema.json`.

---

### 4. Prepare Features

```bash
python scripts/data_prep.py --input data/cert_dataset.csv --output data/features.csv --split
```

**Expected Output**:
```
Loading data from: data/cert_dataset.csv
Loaded 15 rows, 7 columns
Generated X user-day feature rows
✓ Saved features to: data/features.csv
✓ Saved train split: data/features_train.csv
✓ Saved test split: data/features_test.csv
```

---

### 5. Train Isolation Forest (Unsupervised)

```bash
python scripts/train_iso.py --input data/features_train.csv --contamination 0.01
```

**Expected Output**:
```
Training Isolation Forest...
✓ Model training complete
✓ Saved model to: models/iso_model.pkl
✓ Saved scaler to: models/iso_scaler.pkl
```

---

### 6. Train XGBoost (Supervised - Requires Labels)

```bash
python scripts/train_xgb.py --input data/features_train.csv --test_path data/features_test.csv
```

**Expected Output**:
```
Training XGBoost classifier...
✓ Model training complete
Test Set Metrics:
  Accuracy: 0.XX
  ROC-AUC: 0.XX
✓ Saved model to: models/xgb_model.pkl
```

**Note**: This requires a dataset with labels. If your dataset doesn't have labels, skip this step.

---

### 7. Evaluate Models

```bash
python scripts/evaluate.py --test_data data/features_test.csv
```

**Expected Output**:
```
Evaluating XGBoost model...
  Precision: 0.XX
  Recall: 0.XX
✓ Saved summary metrics to: artifacts/summary_metrics.csv
✓ Saved comparison report to: artifacts/model_comparison.md
```

---

### 8. Generate SHAP Explanations (If XGBoost Trained)

```bash
python scripts/explain_xgb_shap.py --model_path models/xgb_model.pkl --test_data data/features_test.csv
```

**Expected Output**:
```
Computing SHAP values...
✓ Saved summary plot to: artifacts/shap_summary.png
```

---

### 9. Start API Server

```bash
# On Windows PowerShell:
bash scripts/run_api.sh

# On Linux/Mac:
chmod +x scripts/run_api.sh
./scripts/run_api.sh
```

**Expected Output**:
```
Starting Insider Threat Detection API...
API will be available at: http://localhost:8000
```

**Keep this terminal open!** The API is now running.

---

### 10. Test API (In New Terminal)

Open a **new terminal window** and run:

```bash
# Activate venv again (if needed)
cd insider-threat
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_prediction_request.json
```

**Expected Output**:
```json
{
  "xgb_prediction": {
    "prediction": 0,
    "probability_anomaly": 0.15,
    "label": "normal"
  },
  "iso_score": -0.123,
  "iso_prediction": "normal"
}
```

---

### 11. Run Tests

```bash
pytest tests/ -v
```

**Expected Output**:
```
tests/test_schema.py::test_detect_column_type_timestamp PASSED
tests/test_data_prep.py::test_engineer_features_basic PASSED
...
```

---

## Complete Command Sequence (Copy-Paste)

For convenience, here's the entire sequence in one block:

```bash
# Setup
cd insider-threat
python -m venv venv
venv\Scripts\activate  # Windows (use source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt

# Data preparation
bash scripts/verify_dataset.sh
python scripts/schema_and_inventory.py data/cert_dataset.csv
python scripts/data_prep.py --input data/cert_dataset.csv --output data/features.csv --split

# Train models
python scripts/train_iso.py --input data/features_train.csv --contamination 0.01
python scripts/train_xgb.py --input data/features_train.csv --test_path data/features_test.csv

# Evaluate
python scripts/evaluate.py --test_data data/features_test.csv
python scripts/explain_xgb_shap.py --model_path models/xgb_model.pkl --test_data data/features_test.csv

# Test
pytest tests/ -v

# Start API (in separate terminal)
bash scripts/run_api.sh
```

---

## Troubleshooting

### "Module not found" errors
- Ensure virtual environment is activated (`(venv)` in prompt)
- Reinstall dependencies: `pip install -r requirements.txt`

### "File not found" errors
- Check you're in the `insider-threat` directory
- Verify dataset exists: `ls data/` or `dir data\`

### API won't start
- Check port 8000 is not in use: `netstat -an | findstr 8000` (Windows)
- Try different port: Edit `scripts/run_api.sh` and change `--port 8000` to `--port 8001`

### Models not found in API
- Train models first (steps 5-6)
- Check models exist: `ls models/` or `dir models\`

---

## Next Steps

After running all commands successfully:

1. **Explore the notebook**: Open `notebooks/exploration.ipynb` in Jupyter
2. **Read the tutorial**: See `docs/tutorial_for_beginners.md` for detailed explanations
3. **Try experiments**: Follow `docs/experiments.md` for hyperparameter tuning
4. **Deploy with Docker**: See `docker/README.md` for containerization

---

## Summary

✅ **What you've accomplished:**
- Set up a Python environment
- Processed raw event data into features
- Trained two ML models (Isolation Forest and XGBoost)
- Evaluated model performance
- Generated explanations (SHAP)
- Deployed a REST API for predictions
- Ran automated tests

🎉 **Congratulations!** You now have a complete, production-ready insider threat detection system!

