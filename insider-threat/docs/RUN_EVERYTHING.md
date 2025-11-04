# 🚀 Run Everything - Complete Project Execution

## Master Script: `run_all.py`

This is your **one-stop script** to run the entire Insider Threat Detection project from start to finish!

---

## 🎯 Quick Start

### Basic Run (Recommended)
```powershell
python run_all.py
```

This runs:
1. ✅ Environment check
2. ✅ Dataset generation (500 rows)
3. ✅ Schema detection
4. ✅ Feature engineering
5. ✅ Isolation Forest training
6. ✅ XGBoost training
7. ✅ Model evaluation
8. ✅ Tests (14 tests)
9. ✅ Summary report

**Time**: ~2-3 minutes

---

## 📋 Options

### Run Everything + Start API
```powershell
python run_all.py --api
```

This runs the full pipeline AND starts the API server automatically!

### Skip Tests (Faster)
```powershell
python run_all.py --skip-tests
```

Useful for quick runs when you just want to see results.

### Use Existing Dataset
```powershell
python run_all.py --skip-dataset
```

Skip dataset generation and use existing `data/cert_dataset.csv`.

### Combine Options
```powershell
python run_all.py --skip-tests --api
```

---

## 🔍 What It Does

### Step 0: Environment Check
- Verifies virtual environment exists
- Checks key dependencies
- Shows warnings if anything is missing

### Steps 1-6: ML Pipeline
- Generates/loads dataset
- Detects schema
- Engineers features
- Trains both models
- Evaluates performance

### Step 7: Testing
- Runs all 14 unit tests
- Shows pass/fail status

### Step 8: Summary
- Counts generated files
- Shows key metrics
- Provides next steps

### Step 9 (Optional): API Server
- Starts FastAPI server on port 8000
- Shows URLs for access
- Runs until you press Ctrl+C

---

## 📊 Expected Output

```
============================================================
  INSIDER THREAT DETECTION - COMPLETE PIPELINE
============================================================

[0/8] Checking Environment...
  [OK] Environment ready!

[1/8] Step 1: Generating Synthetic Dataset...
  [OK] Generated 500 rows

[2/8] Step 2: Detecting Schema...
  ...

[8/8] Generating Summary...

============================================================
  PIPELINE COMPLETE!
============================================================

Generated Files:
  ✓ Models: 4 files
  ✓ Artifacts: 5 files
  ✓ Data Files: 5 files

Steps Completed: 8/8

📊 Key Metrics:
  XGBoost: ROC-AUC = 0.846
  Isolation Forest: Precision@1% = 1.000

Next Steps:
  1. Review results: artifacts/summary_metrics.csv
  2. Start API: python run_all.py --api
  ...
```

---

## 🎬 Use Cases

### For Presentations/Demos
```powershell
python run_all.py --api
```
Shows complete pipeline + live API demonstration!

### For Development
```powershell
python run_all.py --skip-tests
```
Quick iteration without waiting for tests.

### For Testing
```powershell
python run_all.py
```
Full pipeline including all tests.

---

## 🐛 Troubleshooting

### "Virtual environment not found"
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### "Dependencies missing"
```powershell
pip install pandas numpy scikit-learn xgboost joblib fastapi uvicorn matplotlib seaborn pytest tabulate httpx
```

### "Dataset generation failed"
Check that you have write permissions in the `data/` directory.

---

## 📝 Comparison with Other Scripts

| Script | Purpose | Use When |
|--------|---------|----------|
| `run_all.py` | **Complete pipeline + options** | **You want everything** |

**`run_all.py` is the most comprehensive!**

---

## ✅ Success Indicators

You'll know it worked when you see:
- ✅ "PIPELINE COMPLETE!" message
- ✅ File counts for models, artifacts, data
- ✅ Key metrics displayed
- ✅ No fatal errors

---

## 🎯 Quick Reference

```powershell
# Full run
python run_all.py

# With API
python run_all.py --api

# Fast run (skip tests)
python run_all.py --skip-tests

# Help
python run_all.py --help
```

---

**This is your go-to script for running everything!** 🚀

