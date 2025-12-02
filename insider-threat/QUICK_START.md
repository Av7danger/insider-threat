# 🚀 Quick Start Guide

## Complete Demo Setup

### Step 1: Install Dependencies
```powershell
python -m pip install streamlit plotly shap joblib xgboost matplotlib requests pandas numpy scikit-learn
```

### Step 2: Start the Demo

**Option A - Use the Launcher (Recommended):**
```powershell
cd insider-threat
.\start_demo.ps1
```

**Option B - Manual Start:**

Terminal 1 (API Backend):
```powershell
cd insider-threat
python -m uvicorn app.inference_api:app --host 0.0.0.0 --port 8000
```

Terminal 2 (Frontend):
```powershell
cd insider-threat
python -m streamlit run demo_app.py
```

### Step 3: Access the Demo

- **Frontend:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

## Troubleshooting

### If `streamlit` command not found:
Use `python -m streamlit` instead:
```powershell
python -m streamlit run demo_app.py
```

### If packages are missing:
```powershell
python -m pip install -r requirements.txt
python -m pip install streamlit plotly shap
```

## What's Included

✅ **Frontend:** Interactive Streamlit web app
✅ **Backend:** FastAPI REST API
✅ **Models:** XGBoost & Isolation Forest
✅ **Visualizations:** Charts, SHAP explanations
✅ **Sample Data:** Ready to use

---

**Ready to go!** Open http://localhost:8501 after starting the services.

