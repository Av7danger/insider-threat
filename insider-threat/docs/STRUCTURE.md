# Project Directory Structure

This document describes the clean, organized structure of the Insider Threat Detection project.

## Directory Layout

```
insider-threat/
├── README.md                    # Main project README
├── requirements.txt             # Python dependencies
├── pytest.ini                   # pytest configuration
├── run_all.py                   # Master script to run entire pipeline
├── demo_app.py                  # Streamlit demo application (main entry point)
│
├── app/                         # FastAPI application
│   ├── __init__.py
│   └── inference_api.py         # REST API for model inference
│
├── scripts/                     # All executable scripts
│   ├── create_dataset.py        # Generate synthetic dataset
│   ├── schema_and_inventory.py  # Schema detection
│   ├── data_prep.py             # Feature engineering
│   ├── train_iso.py             # Isolation Forest training
│   ├── train_xgb.py             # XGBoost training
│   ├── train_lstm.py            # LSTM training
│   ├── evaluate.py              # Model evaluation
│   ├── explain_xgb_shap.py      # SHAP explainability
│   ├── lstm_infer.py            # LSTM inference helper
│   ├── hparam_search.py         # Hyperparameter search
│   ├── create_release_notes.py  # Release automation
│   ├── save_artifacts.sh        # Artifact collection
│   ├── run_api.sh               # Start FastAPI server
│   ├── run_demo.sh              # Streamlit demo launcher (bash)
│   └── verify_dataset.sh        # Dataset verification
│
├── tests/                       # Unit tests
│   ├── test_schema.py
│   ├── test_data_prep.py
│   ├── test_api_validation.py
│   └── test_demo_app.py
│
├── docs/                        # All documentation
│   ├── STRUCTURE.md             # This file
│   ├── README.md                # Project overview (if exists)
│   ├── QUICK_START.md           # Quick start guide
│   ├── tutorial_for_beginners.md
│   ├── experiments.md
│   ├── release_checklist.md
│   ├── ci.md
│   ├── COMMIT_STYLE.md
│   ├── demo_instructions.md
│   ├── demo_slide_speakers_notes.md
│   ├── PROJECT_SUMMARY.md
│   ├── QUICK_START.md
│   ├── RUN_EVERYTHING.md
│   ├── STRUCTURE.md
│   ├── TASKS.md
│   └── tutorial_for_beginners.md
│
├── notebooks/                   # Jupyter notebooks
│   └── exploration.ipynb
│
├── data/                        # Data files (gitignored)
│   ├── sample_cert_small.csv    # Small sample (tracked)
│   ├── cert_dataset.csv         # Generated dataset
│   ├── features.csv             # Engineered features
│   ├── features_train.csv
│   ├── features_test.csv
│   └── detected_schema.json
│
├── models/                      # Trained models (gitignored)
│   ├── iso_model.pkl
│   ├── iso_scaler.pkl
│   ├── xgb_model.pkl
│   ├── xgb_scaler.pkl
│   └── iso_train_scores.csv
│
├── artifacts/                   # Output artifacts (gitignored)
│   ├── summary_metrics.csv
│   ├── xgb_metrics.json
│   ├── model_comparison.md
│   ├── roc.png
│   ├── xgb_confusion_matrix.png
│   ├── shap_summary_demo.png
│   ├── demo_scores.csv
│   └── shap_force_*.png
│
├── docker/                      # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── README.md
│
├── .github/                     # GitHub workflows
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
│
└── venv/                        # Virtual environment (gitignored)
```

## Key Principles

1. **All scripts in `scripts/`** - No executable scripts in root except `run_all.py` and `demo_app.py`
2. **All documentation in `docs/`** - All `.md` files except `README.md` are in `docs/`
3. **Main entry points in root** - `run_all.py` and `demo_app.py` stay in root for easy access
4. **Generated files gitignored** - Data, models, artifacts, and venv are excluded from git
5. **Clear separation** - Code, tests, docs, config all in separate directories

## Entry Points

### For Users
- **Quick Start**: `docs/QUICK_START.md`
- **Full Pipeline**: `python run_all.py`
- **Interactive Demo**: `streamlit run demo_app.py`
- **API Server**: `bash scripts/run_api.sh`

### For Developers
- **Tests**: `pytest tests/ -v`
- **CI/CD**: `.github/workflows/ci.yml`
- **Documentation**: `docs/` directory

## File Naming Conventions

- **Scripts**: `snake_case.py` or `snake_case.sh`
- **Tests**: `test_*.py`
- **Documentation**: `UPPER_CASE.md` or `snake_case.md`
- **Models**: `*_model.pkl`, `*_scaler.pkl`
- **Data**: `*.csv`, `*.json`

## Migration Notes

The following files were moved/removed during restructuring:
- All `.md` files (except `README.md`) → `docs/`
- `run_demo.sh` → `scripts/` (Streamlit demo launcher)
- Duplicate `generate_synthetic_data.py` removed (use `create_dataset.py`)
- Redundant demo scripts removed (`run_demo.py`, `run_demo.ps1`) - use `run_all.py` instead
- Redundant/outdated docs removed: `FINAL_SUMMARY.md`, `RUN_RESULTS.md`, `QUICK_DEMO.md`, `LIVE_DEMO.md`, `DEMO_GUIDE.md`, `EXECUTION_POLICY_FIX.md`, `VERIFICATION.md`

