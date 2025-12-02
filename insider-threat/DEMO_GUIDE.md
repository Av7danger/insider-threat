# 🚀 Complete Demo Guide - Insider Threat Detection

## Quick Start

### Option 1: Use the PowerShell Launcher (Easiest)
```powershell
cd insider-threat
.\start_demo.ps1
```

This will automatically:
- Start the FastAPI backend
- Launch the Streamlit frontend
- Open your browser automatically

### Option 2: Manual Start

**Terminal 1 - Start API Backend:**
```powershell
cd insider-threat
python -m uvicorn app.inference_api:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Frontend:**
```powershell
cd insider-threat
streamlit run demo_app.py
```

## 🌐 Access Points

- **Frontend (Streamlit):** http://localhost:8501
- **Backend API Docs:** http://localhost:8000/docs
- **API Health Check:** http://localhost:8000/health

## 📱 Using the Frontend

### Step 1: Open the Demo
Navigate to http://localhost:8501 in your browser

### Step 2: Load Data
- Click **"Load sample"** to use the sample dataset, OR
- Upload your own CSV file using the file uploader

### Step 3: Configure Settings
- **Mode:** Choose "Use FastAPI" (recommended) or "Local models"
- **Top k anomalies:** Adjust slider to show top N anomalies (default: 10)
- **Show SHAP explanations:** Check to enable model explanations

### Step 4: Run Analysis
Click the **"Run Inference"** button

### Step 5: View Results
The demo will show:
- **Top anomalous user-days table** with scores and rankings
- **Interactive bar charts** showing anomaly scores
- **ROC curves** (if labels are available)
- **SHAP explanations** showing which features contributed to predictions
- **Download button** to save results as CSV

## 🎨 Frontend Features

### Interactive Visualizations
- **Anomaly Score Charts:** Bar charts showing top-k anomalies
- **ROC Curves:** Model performance visualization
- **SHAP Summary Plots:** Feature importance visualization
- **Force Plots:** Individual prediction explanations

### Data Management
- **CSV Upload:** Drag-and-drop file upload
- **Sample Data:** One-click sample data loading
- **Results Export:** Download predictions as CSV

### Model Options
- **FastAPI Mode:** Uses trained models via REST API (faster, recommended)
- **Local Models:** Uses models directly (works offline)

## 🔌 API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

### Single Prediction
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "total_events": 150.0,
  "unique_src_ip": 3.0,
  "unique_dst_ip": 10.0,
  "distinct_files": 45.0,
  "avg_success": 0.95,
  "start_hour": 9.0,
  "end_hour": 17.0,
  "peak_hour": 14.0
}
```

### Batch Prediction
```bash
POST http://localhost:8000/predict_batch
Content-Type: application/json

[
  {
    "total_events": 150.0,
    "unique_src_ip": 3.0,
    ...
  },
  ...
]
```

## 📊 Sample Data Format

Your CSV should have columns like:
- `user` - User identifier
- `date` or `timestamp` - Date/time of activity
- `src_ip` - Source IP address
- `dst_ip` - Destination IP address  
- `file_path` - File accessed
- `success` - Success status (0/1, true/false, etc.)
- `label` - (Optional) Ground truth label

## 🎯 Example Workflow

1. **Start Services:**
   ```powershell
   .\start_demo.ps1
   ```

2. **Open Browser:** http://localhost:8501

3. **Load Sample Data:** Click "Load sample" button

4. **Run Analysis:** Click "Run Inference"

5. **Explore Results:**
   - Review top anomalies in the table
   - Check visualizations
   - View SHAP explanations
   - Download results

6. **Try Your Own Data:**
   - Upload your CSV file
   - Run inference
   - Compare results

## 🛠️ Troubleshooting

### Frontend won't load
- Check if Streamlit is running: `Get-Process python`
- Try accessing http://localhost:8501 directly
- Check firewall settings

### API not responding
- Verify API is running: `Invoke-RestMethod http://localhost:8000/health`
- Check if port 8000 is available
- Review API logs for errors

### Models not found
- Ensure you've run the training pipeline: `python run_all.py`
- Check `models/` directory for `.pkl` files
- Use "Local models" mode as fallback

### SHAP errors
- Install SHAP: `pip install shap`
- Some features may not work without XGBoost model
- Check console for detailed error messages

## 📁 File Locations

- **Models:** `models/iso_model.pkl`, `models/xgb_model.pkl`
- **Sample Data:** `data/sample_cert_small.csv`
- **Results:** `artifacts/demo_scores.csv`
- **SHAP Visualizations:** `artifacts/shap_*.png`

## 🎓 Learning Resources

- **API Documentation:** http://localhost:8000/docs (interactive)
- **Project README:** `README.md`
- **Demo Instructions:** `docs/demo_instructions.md`
- **Tutorial:** `docs/tutorial_for_beginners.md`

## ✨ Tips

- Use "FastAPI" mode for best performance
- Enable SHAP explanations to understand model decisions
- Export results for further analysis
- Try different top-k values to see more/fewer anomalies
- Compare XGBoost and Isolation Forest predictions

---

**Enjoy exploring the Insider Threat Detection system!** 🔍

