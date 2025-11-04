# Streamlit Demo Instructions

Complete step-by-step guide for running the Insider Threat Detection Streamlit demo.

## Quick Start

The demo app provides an interactive interface to upload activity data, run ML inference, and visualize results with SHAP explanations.

---

## Prerequisites

### 1. Create and Activate Virtual Environment

**Unix/Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

Install base requirements plus demo extras:

```bash
pip install -r requirements.txt
pip install streamlit plotly requests shap joblib xgboost matplotlib
```

**Note:** If you encounter issues with SHAP, ensure you have:
- `numba` (for SHAP performance)
- `matplotlib` backend properly configured

---

## Running the Demo

### Option 1: With FastAPI Backend (Recommended)

This uses the trained models via FastAPI for better performance.

**Terminal 1 - Start API:**
```bash
bash scripts/run_api.sh
```

**Terminal 2 - Start Streamlit:**
```bash
streamlit run demo_app.py
```

The Streamlit app will automatically open in your browser at `http://localhost:8501`.

**In the Streamlit app:**
1. Select **"Use FastAPI"** mode (default)
2. Click **"Load sample"** or upload your own CSV
3. Click **"Run Inference"**
4. View results, visualizations, and SHAP explanations

### Option 2: Local Models Only

Run without the FastAPI backend (uses models directly):

```bash
streamlit run demo_app.py
```

**In the Streamlit app:**
1. Select **"Local models"** mode
2. Click **"Load sample"** or upload your own CSV
3. Click **"Run Inference"**
4. View results

**Note:** If models are not found, a quick IsolationForest will be trained automatically.

### Option 3: Using the Launcher Script

**Unix/Linux/Mac:**
```bash
bash scripts/run_demo.sh
```

The script will:
- Ask if you want to start FastAPI backend
- Start Streamlit automatically
- Handle cleanup when you stop

**Windows PowerShell (manual):**
```powershell
# Terminal 1:
bash scripts/run_api.sh

# Terminal 2:
streamlit run demo_app.py
```

---

## Generating Sample Data

If you need to generate a sample dataset:

```bash
python scripts/create_dataset.py
```

This creates `data/cert_dataset.csv` with 500 rows of synthetic activity data.

---

## File Locations

After running the demo, you'll find:

### Output Files

- **`artifacts/demo_scores.csv`** - Complete results with all scores and predictions
  - Columns: user, date, iso_score, xgb_prob, xgb_pred, and all feature columns

- **`artifacts/shap_summary_demo.png`** - SHAP summary plot showing feature importance
  - Generated when SHAP explanations are enabled

- **`artifacts/shap_force_<index>.png`** - Individual SHAP force plots for top anomalies
  - Generated for top 5 anomalies when SHAP is enabled

### Input Files

- **`data/sample_cert_small.csv`** - Small sample dataset (15 rows)
- **`data/cert_dataset.csv`** - Full synthetic dataset (500 rows, generated)

---

## Expected UI Layout

### Sidebar (Left)
- **Mode toggle**: Choose FastAPI or Local models
- **File uploader**: Upload CSV or click "Load sample"
- **Top k slider**: Number of top anomalies to display (1-50)
- **SHAP checkbox**: Enable/disable SHAP explanations
- **Run Inference button**: Process the data

### Main Content Area

**Left Column (60%):**
- **Uploaded data preview**: First 10 rows of uploaded CSV
- **Top anomalous user-days table**: 
  - Columns: rank, user, date, iso_score, xgb_prob, reason_short
  - Paginated for easy navigation
- **Download results button**: Download CSV with all scores

**Right Column (40%):**
- **Top-k anomaly scores chart**: Horizontal bar chart (Plotly)
  - Users on y-axis, scores on x-axis
  - Color-coded by score (blue to red)
- **ROC curve**: If labels exist in data, shows model performance
- **SHAP explanations**: 
  - Summary plot image
  - Per-row textual explanations for top anomalies

### Footer
- Status message showing where results were saved

---

## UI Mockup (Textual Description)

```
┌─────────────────────────────────────────────────────────────┐
│         Insider Threat Detection — Live Demo                 │
├──────────────┬──────────────────────────────────────────────┤
│              │                                              │
│  SIDEBAR     │  MAIN CONTENT AREA                           │
│              │                                              │
│  Insider     │  ┌──────────────────────────────────────┐  │
│  Threat      │  │ Uploaded data preview                 │  │
│  Demo        │  │ [Table with 10 rows]                  │  │
│              │  └──────────────────────────────────────┘  │
│  Mode:       │                                              │
│  ○ FastAPI  │  ┌──────────────────────────────────────┐  │
│  ○ Local    │  │ Top anomalous user-days              │  │
│              │  │ rank│user │date      │score│prob    │  │
│  [Upload]   │  │ 1   │u003 │2020-01-01│0.85 │0.92    │  │
│              │  │ 2   │u005 │2020-01-02│0.78 │0.85    │  │
│  [Load      │  │ ...                                    │  │
│   sample]   │  └──────────────────────────────────────┘  │
│              │                                              │
│  Top k: 10  │  ┌──────────────┐  ┌─────────────────┐    │
│              │  │ Bar Chart    │  │ ROC Curve       │    │
│  ☑ SHAP     │  │ (Top scores) │  │ (if labels)     │    │
│              │  └──────────────┘  └─────────────────┘    │
│  [Run       │                                              │
│  Inference] │  ┌──────────────────────────────────────┐  │
│              │  │ SHAP Summary Plot                    │  │
│              │  │ [Image showing feature importance]    │  │
│              │  └──────────────────────────────────────┘  │
│              │                                              │
│  How to     │  ✓ Saved results to artifacts/demo_scores.csv│
│  demo:      │                                              │
│  [Expand]   │                                              │
└──────────────┴──────────────────────────────────────────────┘
```

---

## Troubleshooting

### FastAPI Not Reachable

**Problem:** Streamlit shows "FastAPI not reachable — switching to local models"

**Solutions:**
1. **Check if API is running:**
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"ok","service":"insider-threat-detection"}`

2. **Start the API:**
   ```bash
   bash scripts/run_api.sh
   ```
   Wait a few seconds for it to start, then try again in Streamlit.

3. **Check port conflicts:**
   - Ensure port 8000 is not used by another application
   - Try changing the port in `scripts/run_api.sh` if needed

4. **Use Local models mode:**
   - Select "Local models" in the Streamlit sidebar
   - This bypasses the API requirement

---

### Missing Models

**Problem:** "XGBoost model not found; XGBoost results and SHAP will be skipped."

**Solutions:**
1. **Train models first:**
   ```bash
   python scripts/train_iso.py --input data/features_train.csv --contamination 0.01
   python scripts/train_xgb.py --input data/features_train.csv --test_path data/features_test.csv
   ```

2. **Check model files exist:**
   ```bash
   ls models/*.pkl
   ```
   Should show: `iso_model.pkl`, `iso_scaler.pkl`, `xgb_model.pkl`, `xgb_scaler.pkl`

3. **Use sample data:**
   ```bash
   python scripts/create_dataset.py
   python scripts/data_prep.py --input data/cert_dataset.csv --output data/features.csv --split
   ```

4. **Fallback:** The app will train a quick IsolationForest if no models are found, but XGBoost and SHAP will be unavailable.

---

### SHAP Plotting Backend Errors

**Problem:** Error like "matplotlib backend not available" or "No display name"

**Solutions:**
1. **Set matplotlib backend (already done in code):**
   The demo app automatically sets `matplotlib.use('Agg')` for headless environments.

2. **If still having issues, verify installation:**
   ```bash
   pip install matplotlib --upgrade
   ```

3. **Check headless environment:**
   ```python
   import matplotlib
   print(matplotlib.get_backend())  # Should show 'Agg' or similar
   ```

4. **Disable SHAP temporarily:**
   - Uncheck "Show SHAP explanations" in the Streamlit sidebar
   - The app will still show scores and visualizations

---

### File Upload Errors

**Problem:** CSV upload fails or shows parsing errors

**Solutions:**
1. **Check CSV format:**
   - Ensure it has columns: user, date (or timestamp), and optionally: src_ip, dst_ip, file_path, success, label
   - Date column should be in a recognizable format (YYYY-MM-DD HH:MM:SS, etc.)

2. **Try sample data first:**
   - Click "Load sample" to verify the app works
   - Then try your own data

3. **Check file encoding:**
   - Ensure CSV is UTF-8 encoded
   - Avoid special characters in column names

---

### Port Already in Use

**Problem:** "Address already in use" when starting Streamlit or API

**Solutions:**
1. **Find and kill existing process:**
   ```bash
   # Unix/Linux/Mac:
   lsof -ti:8501 | xargs kill  # Streamlit
   lsof -ti:8000 | xargs kill  # FastAPI
   
   # Windows:
   netstat -ano | findstr :8501
   taskkill /PID <PID> /F
   ```

2. **Use different ports:**
   ```bash
   streamlit run demo_app.py --server.port 8502
   ```

---

## Tips for Best Experience

1. **Start with sample data:** Use "Load sample" to see the app in action first
2. **Enable SHAP:** Check the SHAP checkbox for detailed explanations
3. **Adjust top-k:** Use the slider to see more or fewer top anomalies
4. **Download results:** Save the CSV for further analysis
5. **Try both modes:** Compare FastAPI vs Local models performance

---

## Next Steps

- **Explore results:** Open `artifacts/demo_scores.csv` in Excel or Python
- **Customize:** Modify the demo app code to add new visualizations
- **Integrate:** Use the FastAPI endpoint in your own applications
- **Learn more:** See `docs/tutorial_for_beginners.md` for ML concepts

---

**Happy demoing!** 🚀

