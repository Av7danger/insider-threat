# Tutorial for Beginners

Welcome! This step-by-step tutorial will walk you through the entire insider threat detection project, from setup to running predictions. Think of this as a teacher explaining each step.

## Prerequisites

- Python 3.10 or higher installed
- Basic command-line familiarity
- No prior ML experience needed (we'll explain as we go!)

## Step 1: Setting Up the Environment

### What We're Doing

We're creating an isolated Python environment so our project dependencies don't conflict with other projects on your computer.

### Commands

```bash
# Create a virtual environment (think of it as a clean workspace)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### Expected Output

You should see packages being installed. When done, you'll see `(venv)` at the start of your command prompt.

### Troubleshooting

**Problem**: `python` command not found  
**Solution**: Try `python3` instead, or install Python from python.org

**Problem**: `pip` not found  
**Solution**: Python usually comes with pip. Try `python -m pip install -r requirements.txt`

---

## Step 2: Verifying the Dataset

### What We're Doing

We're checking that the dataset file exists and looks correct before processing it.

### Commands

```bash
# On Windows PowerShell:
bash scripts/verify_dataset.sh
# Or on Linux/Mac:
chmod +x scripts/verify_dataset.sh
./scripts/verify_dataset.sh
```

### Expected Output

```
✓ Dataset file found: data/cert_dataset.csv
Rows: 10,000
Columns: 15
Column names:
  1. user
  2. date
  3. src_ip
  ...
```

### Troubleshooting

**Problem**: File not found  
**Solution**: Place your dataset CSV at `data/cert_dataset.csv`, or use the sample file at `data/sample_cert_small.csv`

---

## Step 3: Understanding Your Data (Schema Detection)

### What We're Doing

We're analyzing the dataset to understand what columns it has, what types of data they contain, and whether there are missing values.

### Commands

```bash
python scripts/schema_and_inventory.py data/cert_dataset.csv
```

### Expected Output

You'll see a summary showing:
- Total rows and columns
- Column types (timestamp, numeric, categorical, etc.)
- Missing value counts
- Sample values from each column

### What You'll Learn

- **Timestamps**: Dates and times that need special parsing
- **Numeric columns**: Numbers that can be used directly
- **Categorical columns**: Text values with limited options
- **Missing values**: Gaps in data that need handling

---

## Step 4: Preparing Features

### What We're Doing

We're converting raw event logs into "features" - numbers that describe user behavior. Instead of thousands of individual events, we create one row per user per day with summary statistics.

### Why This Matters

Machine learning models need structured data. We aggregate:
- How many events did the user have? (total_events)
- How many different IPs did they use? (unique_src_ip)
- How many files did they access? (distinct_files)
- What time did they work? (start_hour, end_hour)

### Commands

```bash
python scripts/data_prep.py --input data/cert_dataset.csv --output data/features.csv --split
```

### Expected Output

```
Loading data from: data/cert_dataset.csv
Loaded 10,000 rows, 15 columns
Using columns:
  Timestamp: date
  User: user
  Label: label
Aggregating features by user and date...
Generated 500 user-day feature rows
✓ Saved features to: data/features.csv
✓ Saved train split: data/features_train.csv (400 rows)
✓ Saved test split: data/features_test.csv (100 rows)
```

### What You'll Learn

- **Feature engineering**: Creating useful numbers from raw data
- **Aggregation**: Summarizing many events into single values
- **Train/test split**: Separating data for training vs. evaluation

---

## Step 5: Training the Isolation Forest Model

### What We're Doing

We're training an "unsupervised" model that learns what "normal" looks like without needing labels. It flags anything unusual.

### Commands

```bash
python scripts/train_iso.py --input data/features_train.csv --contamination 0.01
```

### Expected Output

```
Training Isolation Forest...
  Contamination: 0.01 (expecting 1.0% anomalies)
  Number of trees: 100
✓ Model training complete
Training Summary:
  Total samples: 400
  Anomalies detected: 4 (1.00%)
```

### What You'll Learn

- **Unsupervised learning**: Finding patterns without labels
- **Anomaly detection**: Identifying unusual behavior
- **Contamination rate**: Expected percentage of anomalies

### Troubleshooting

**Problem**: Model training is slow  
**Solution**: Reduce `--n_estimators` (fewer trees = faster but less accurate)

---

## Step 6: Training the XGBoost Model (If You Have Labels)

### What We're Doing

We're training a "supervised" model that learns from examples labeled as "normal" or "anomalous". This requires a dataset with labels.

### Commands

```bash
python scripts/train_xgb.py --input data/features_train.csv --test_path data/features_test.csv
```

### Expected Output

```
Training XGBoost classifier...
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 5
✓ Model training complete
Test Set Metrics:
  Accuracy: 0.95
  ROC-AUC: 0.92
  Precision: 0.88
  Recall: 0.75
  F1-Score: 0.81
```

### What You'll Learn

- **Supervised learning**: Learning from labeled examples
- **Classification**: Predicting categories (normal vs. anomalous)
- **Evaluation metrics**: How to measure model performance

### Troubleshooting

**Problem**: "Label column not found"  
**Solution**: Your dataset doesn't have labels. Use Isolation Forest (Step 5) instead, or add a label column.

**Problem**: Poor performance (low accuracy)  
**Solution**: Try adjusting hyperparameters (see `docs/experiments.md`) or adding more features

---

## Step 7: Evaluating Models

### What We're Doing

We're comparing how well our models perform on test data they've never seen before.

### Commands

```bash
python scripts/evaluate.py --test_data data/features_test.csv
```

### Expected Output

```
Evaluating XGBoost model...
  Precision: 0.8800
  Recall: 0.7500
  F1-Score: 0.8100
  ROC-AUC: 0.9200
Evaluating Isolation Forest model...
  Precision@1%: 0.8500
  Recall@1%: 0.6000
✓ Saved summary metrics to: artifacts/summary_metrics.csv
✓ Saved comparison report to: artifacts/model_comparison.md
✓ Saved ROC curves to: artifacts/roc.png
```

### What You'll Learn

- **Precision**: How many flagged users are actually anomalous?
- **Recall**: How many actual anomalies did we catch?
- **ROC curve**: Visual comparison of model performance

---

## Step 8: Understanding Predictions (SHAP Explanations)

### What We're Doing

We're generating explanations that show WHY the model flagged a user as anomalous.

### Commands

```bash
python scripts/explain_xgb_shap.py --model_path models/xgb_model.pkl --test_data data/features_test.csv
```

### Expected Output

```
Computing SHAP values for 200 samples...
✓ SHAP values computed
✓ Saved summary plot to: artifacts/shap_summary.png
✓ Saved force plots for top 5 anomalies
```

### What You'll Learn

- **Feature importance**: Which behaviors matter most?
- **Individual explanations**: Why was THIS user flagged?
- **Trust and transparency**: Understanding model decisions

---

## Step 9: Starting the API

### What We're Doing

We're starting a web service that can make predictions on new data in real-time.

### Commands

```bash
# On Windows PowerShell:
bash scripts/run_api.sh
# Or on Linux/Mac:
./scripts/run_api.sh
```

### Expected Output

```
Starting Insider Threat Detection API...
API will be available at: http://localhost:8000
Health check: http://localhost:8000/health
API docs: http://localhost:8000/docs
```

### Testing the API

Open a new terminal and run:

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_prediction_request.json
```

### What You'll Learn

- **REST APIs**: How to expose ML models as web services
- **Real-time inference**: Making predictions on demand
- **Integration**: How other systems can use your model

---

## Common Gotchas and Troubleshooting

### Gotcha 1: Missing Timestamp Formats

**Problem**: Script fails to parse dates  
**Solution**: Check your timestamp format. Common formats are supported, but you may need to modify `parse_timestamp_robust()` in `data_prep.py`

### Gotcha 2: Class Imbalance

**Problem**: Model predicts "normal" for everything  
**Solution**: Insider threats are rare! Use class weights or adjust the contamination rate for Isolation Forest

### Gotcha 3: Model File Not Found

**Problem**: API says "No models available"  
**Solution**: Train models first (Steps 5-6), then start the API

### Gotcha 4: Out of Memory

**Problem**: Training fails with memory error  
**Solution**: 
- Use smaller dataset sample
- Reduce batch size (for LSTM)
- Process data in chunks

### Gotcha 5: Slow Training

**Problem**: Models take forever to train  
**Solution**:
- Use fewer estimators (trees)
- Use GPU for LSTM (if available)
- Train on a sample of data first

---

## Next Learning Steps

Now that you've run the full pipeline, here's what to explore next:

1. **Visualization**: Create plots showing feature distributions, anomaly scores over time
2. **Streaming**: Process events in real-time instead of batch processing
3. **Alerting**: Set up notifications when anomalies are detected
4. **Feature Engineering**: Add new features (weekend detection, file type analysis)
5. **Hyperparameter Tuning**: Improve model performance (see `docs/experiments.md`)

---

## Getting Help

If you're stuck:

1. Check error messages carefully - they often tell you what's wrong
2. Review the relevant script's docstring for usage examples
3. Check `TASKS.md` for known issues and solutions
4. Look at test files to see expected usage patterns

Remember: Learning ML is a journey! Don't worry if things don't work perfectly the first time. 🚀

