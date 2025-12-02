"""
Insider Threat Detection - Streamlit Demo Application

This Streamlit app provides an interactive demo for the Insider Threat Detection system.
It allows users to upload activity data, run inference using either FastAPI backend
or local models, and visualize results with SHAP explanations.

Installation:
    pip install streamlit plotly shap joblib xgboost matplotlib requests pandas numpy scikit-learn

Usage:
    streamlit run demo_app.py

Or with backend:
    Terminal 1: bash scripts/run_api.sh
    Terminal 2: streamlit run demo_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import requests
import json
import traceback
from typing import Optional, Tuple, Dict, List
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Insider Threat Detection — Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color palette
PRIMARY_COLOR = "#0B84FF"
WARNING_COLOR = "#FF6B6B"
TABLE_HEADER_COLOR = "#F3F4F6"

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'features_df' not in st.session_state:
    st.session_state.features_df = None
if 'use_api' not in st.session_state:
    st.session_state.use_api = True
if 'api_reachable' not in st.session_state:
    st.session_state.api_reachable = None

def parse_timestamp_robust(series):
    """
    Parse timestamps supporting multiple common formats.
    
    This handles:
    - ISO format: 2010-01-01 10:00:00
    - Date only: 2010-01-01
    - Unix timestamp (seconds or milliseconds)
    - Custom formats common in log files
    """
    # Try standard pandas parsing first
    try:
        return pd.to_datetime(series)
    except:
        pass
    
    # Try Unix timestamp
    try:
        if series.dropna().astype(float).max() > 1e12:
            return pd.to_datetime(series.astype(float) / 1000, unit='s')
        else:
            return pd.to_datetime(series.astype(float), unit='s')
    except:
        pass
    
    # Try common date formats
    formats = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y']
    for fmt in formats:
        try:
            return pd.to_datetime(series, format=fmt, errors='coerce')
        except:
            continue
    
    raise ValueError(f"Could not parse timestamps from series")

def find_column_by_keywords(df, keywords):
    """Find column name by keywords (case-insensitive)."""
    for col in df.columns:
        if any(keyword in col.lower() for keyword in keywords):
            return col
    return None

def engineer_features(df):
    """
    Engineer features from raw event data with flexible column detection.
    
    This function automatically detects columns and aggregates events into
    per-user-per-day features suitable for ML models.
    
    Args:
        df: Raw event dataframe with columns like user, date, IPs, files, etc.
    
    Returns:
        DataFrame with columns: user, date, total_events, unique_src_ip,
        unique_dst_ip, distinct_files, avg_success, start_hour, end_hour, peak_hour
    """
    # Auto-detect columns
    user_col = find_column_by_keywords(df, ['user', 'id', 'username', 'employee'])
    if not user_col:
        user_col = df.columns[0]  # Fallback to first column
    
    timestamp_col = find_column_by_keywords(df, ['date', 'time', 'timestamp', 'datetime'])
    if not timestamp_col:
        # Try to detect by parsing
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(10))
                timestamp_col = col
                break
            except:
                continue
    
    if not timestamp_col:
        raise ValueError("Could not find timestamp column. Please ensure your CSV has a date/time column.")
    
    label_col = find_column_by_keywords(df, ['label', 'target', 'anomaly', 'malicious'])
    
    # Parse timestamps
    df['_parsed_timestamp'] = parse_timestamp_robust(df[timestamp_col])
    df['_date'] = df['_parsed_timestamp'].dt.date
    
    # Group by user and date
    grouped = df.groupby([user_col, '_date'])
    
    features_list = []
    
    for (user, date), group in grouped:
        feature_row = {
            'user': user,
            'date': date
        }
        
        # Total events
        feature_row['total_events'] = len(group)
        
        # Unique source IPs
        ip_cols = [c for c in group.columns if 'ip' in c.lower() or 'src' in c.lower()]
        if ip_cols:
            feature_row['unique_src_ip'] = group[ip_cols[0]].nunique()
        else:
            feature_row['unique_src_ip'] = 0
        
        # Unique destination IPs
        dst_cols = [c for c in group.columns if 'dst' in c.lower() or 'destination' in c.lower()]
        if dst_cols:
            feature_row['unique_dst_ip'] = group[dst_cols[0]].nunique()
        else:
            feature_row['unique_dst_ip'] = 0
        
        # Distinct files
        file_cols = [c for c in group.columns if 'file' in c.lower() or 'path' in c.lower()]
        if file_cols:
            feature_row['distinct_files'] = group[file_cols[0]].nunique()
        else:
            feature_row['distinct_files'] = 0
        
        # Average success rate
        success_cols = [c for c in group.columns if 'success' in c.lower() or 'status' in c.lower()]
        if success_cols:
            success_values = group[success_cols[0]].astype(str).str.lower()
            success_binary = success_values.isin(['1', 'true', 'success', 'succeeded', 'ok', '200']).astype(int)
            feature_row['avg_success'] = success_binary.mean()
        else:
            feature_row['avg_success'] = 1.0
        
        # Time-based features
        hours = group['_parsed_timestamp'].dt.hour
        feature_row['start_hour'] = hours.min()
        feature_row['end_hour'] = hours.max()
        feature_row['peak_hour'] = hours.mode().iloc[0] if len(hours.mode()) > 0 else hours.median()
        
        # Label if present
        if label_col:
            feature_row['label'] = int(group[label_col].max())
        
        features_list.append(feature_row)
    
    features_df = pd.DataFrame(features_list)
    
    # Handle missing values
    count_cols = ['total_events', 'unique_src_ip', 'unique_dst_ip', 'distinct_files']
    for col in count_cols:
        if col in features_df.columns:
            features_df[col].fillna(0, inplace=True)
    
    continuous_cols = ['avg_success', 'start_hour', 'end_hour', 'peak_hour']
    for col in continuous_cols:
        if col in features_df.columns:
            features_df[col].fillna(features_df[col].median(), inplace=True)
    
    return features_df

def call_api_batch(features_df):
    """
    Call FastAPI /predict_batch endpoint with batch of features.
    
    Args:
        features_df: DataFrame with feature columns
    
    Returns:
        List of prediction responses or None if API unreachable
    """
    api_url = "http://localhost:8000/predict_batch"
    
    # Prepare feature columns in correct order
    feature_cols = [
        'total_events', 'unique_src_ip', 'unique_dst_ip', 'distinct_files',
        'avg_success', 'start_hour', 'end_hour', 'peak_hour'
    ]
    
    # Convert to list of dictionaries for API
    requests_list = []
    for _, row in features_df.iterrows():
        req = {
            'total_events': float(row['total_events']),
            'unique_src_ip': float(row['unique_src_ip']),
            'unique_dst_ip': float(row['unique_dst_ip']),
            'distinct_files': float(row['distinct_files']),
            'avg_success': float(row['avg_success']),
            'start_hour': float(row['start_hour']),
            'end_hour': float(row['end_hour']),
            'peak_hour': float(row['peak_hour'])
        }
        requests_list.append(req)
    
    try:
        response = requests.post(
            api_url,
            json=requests_list,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None  # Endpoint doesn't exist
        return None
    except Exception:
        return None

def load_local_models():
    """
    Load local models from models/ directory.
    
    Returns:
        Tuple of (xgb_model, xgb_scaler, iso_model, iso_scaler) or None for missing models
    """
    xgb_model = None
    xgb_scaler = None
    iso_model = None
    iso_scaler = None
    
    # Load XGBoost
    xgb_path = Path('models/xgb_model.pkl')
    xgb_scaler_path = Path('models/xgb_scaler.pkl')
    if xgb_path.exists() and xgb_scaler_path.exists():
        try:
            xgb_model = joblib.load(xgb_path)
            xgb_scaler = joblib.load(xgb_scaler_path)
        except Exception as e:
            st.warning(f"Could not load XGBoost model: {e}")
    
    # Load Isolation Forest
    iso_path = Path('models/iso_model.pkl')
    iso_scaler_path = Path('models/iso_scaler.pkl')
    if iso_path.exists() and iso_scaler_path.exists():
        try:
            iso_model = joblib.load(iso_path)
            iso_scaler = joblib.load(iso_scaler_path)
        except Exception as e:
            st.warning(f"Could not load Isolation Forest model: {e}")
    
    return xgb_model, xgb_scaler, iso_model, iso_scaler

def score_with_local_models(features_df, xgb_model, xgb_scaler, iso_model, iso_scaler):
    """
    Score features using local models.
    
    If no models are available, trains a quick IsolationForest as fallback.
    
    Args:
        features_df: DataFrame with features
        xgb_model, xgb_scaler: XGBoost model and scaler (or None)
        iso_model, iso_scaler: Isolation Forest model and scaler (or None)
    
    Returns:
        DataFrame with scores added
    """
    results_df = features_df.copy()
    
    feature_cols = [
        'total_events', 'unique_src_ip', 'unique_dst_ip', 'distinct_files',
        'avg_success', 'start_hour', 'end_hour', 'peak_hour'
    ]
    
    X = features_df[feature_cols].values
    
    # Isolation Forest scoring
    if iso_model is not None and iso_scaler is not None:
        X_scaled = iso_scaler.transform(X)
        scores = iso_model.score_samples(X_scaled)
        # Invert scores so higher = more anomalous (original scores are lower for anomalies)
        results_df['iso_score'] = -scores  # Higher score = more anomalous
        results_df['iso_prediction'] = iso_model.predict(X_scaled)
    else:
        # Train quick fallback
        st.info("Training quick IsolationForest fallback (this may take a moment)...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        iso_fallback = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        iso_fallback.fit(X_scaled)
        scores = iso_fallback.score_samples(X_scaled)
        results_df['iso_score'] = -scores
        results_df['iso_prediction'] = iso_fallback.predict(X_scaled)
    
    # XGBoost scoring
    if xgb_model is not None and xgb_scaler is not None:
        X_scaled = xgb_scaler.transform(X)
        try:
            predictions = xgb_model.predict(X_scaled)
            probabilities = xgb_model.predict_proba(X_scaled)
            results_df['xgb_pred'] = predictions
            results_df['xgb_prob'] = probabilities[:, 1]  # Probability of anomaly
        except Exception as e:
            st.warning(f"XGBoost prediction error: {e}")
            results_df['xgb_pred'] = None
            results_df['xgb_prob'] = None
    else:
        results_df['xgb_pred'] = None
        results_df['xgb_prob'] = None
    
    return results_df

def compute_shap_and_save(features_df, xgb_model, xgb_scaler, top_k_indices):
    """
    Compute SHAP values and generate visualizations.
    
    Args:
        features_df: DataFrame with features
        xgb_model: Trained XGBoost model
        xgb_scaler: Feature scaler
        top_k_indices: Indices of top-k anomalies to explain (from original features_df)
    
    Returns:
        Tuple of (shap_values, feature_names, summary_image_path, shap_values_full)
    """
    feature_cols = [
        'total_events', 'unique_src_ip', 'unique_dst_ip', 'distinct_files',
        'avg_success', 'start_hour', 'end_hour', 'peak_hour'
    ]
    
    X = features_df[feature_cols].values
    X_scaled = xgb_scaler.transform(X)
    
    # Use subset for faster computation (max 200 samples)
    n_samples = min(200, len(X_scaled))
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(X_scaled), n_samples, replace=False)
    X_sample = X_scaled[sample_indices]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values_sample = explainer.shap_values(X_sample)
    
    # Handle binary classification (shap_values might be list)
    if isinstance(shap_values_sample, list):
        shap_values_sample = shap_values_sample[1]  # Use positive class
    
    # Compute SHAP for all data (for per-row explanations)
    shap_values_full = explainer.shap_values(X_scaled)
    if isinstance(shap_values_full, list):
        shap_values_full = shap_values_full[1]
    
    # Generate summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_sample, X_sample, feature_names=feature_cols, show=False)
    summary_path = Path('artifacts/shap_summary_demo.png')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate force plots for top-k anomalies
    for idx, orig_idx in enumerate(top_k_indices[:5]):  # Top 5 for visualization
        if orig_idx < len(X_scaled):
            plt.figure(figsize=(12, 4))
            shap.force_plot(
                explainer.expected_value,
                shap_values_full[orig_idx],
                X_scaled[orig_idx],
                feature_names=feature_cols,
                matplotlib=True,
                show=False
            )
            force_path = Path(f'artifacts/shap_force_{idx}.png')
            plt.tight_layout()
            plt.savefig(force_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    return shap_values_sample, feature_cols, str(summary_path), shap_values_full

def get_top_features_text(shap_values_row, feature_names, top_n=3):
    """Extract top contributing features for textual explanation."""
    contribs = [(name, abs(val)) for name, val in zip(feature_names, shap_values_row)]
    contribs.sort(key=lambda x: x[1], reverse=True)
    
    explanations = []
    for name, val in contribs[:top_n]:
        direction = "increases" if val > 0 else "decreases"
        explanations.append(f"{name} {direction} anomaly score")
    
    return "; ".join(explanations)

def render_results(results_df, top_k, show_shap, xgb_model, xgb_scaler):
    """Render results table and visualizations."""
    # Sort by iso_score (higher = more anomalous)
    results_sorted = results_df.sort_values('iso_score', ascending=False)
    top_k_results = results_sorted.head(top_k).copy()
    
    # Create left and right columns
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.subheader("Top anomalous user-days")
        
        # Prepare table data
        table_data = []
        for idx, (_, row) in enumerate(top_k_results.iterrows(), 1):
            table_data.append({
                'rank': idx,
                'user': str(row['user']),
                'date': str(row['date']),
                'iso_score': f"{row['iso_score']:.4f}",
                'xgb_prob': f"{row['xgb_prob']:.4f}" if pd.notna(row.get('xgb_prob')) else "N/A",
                'reason_short': f"High activity: {row['total_events']} events, {row['distinct_files']} files"
            })
        
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        
        # Download button
        output_path = Path('artifacts/demo_scores.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        with open(output_path, 'rb') as f:
            st.download_button(
                label="Download results (CSV)",
                data=f.read(),
                file_name="demo_scores.csv",
                mime="text/csv"
            )
    
    with col2:
        # Top-k anomaly scores bar chart
        st.subheader("Top-k anomaly scores")
        top_k_for_chart = top_k_results.head(min(top_k, 20))  # Limit to 20 for readability
        fig = px.bar(
            top_k_for_chart,
            x='iso_score',
            y='user',
            orientation='h',
            color='iso_score',
            color_continuous_scale=['#0B84FF', '#FF6B6B'],
            labels={'iso_score': 'Anomaly Score', 'user': 'User'},
            height=max(400, len(top_k_for_chart) * 30)
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC curve if labels exist
        if 'label' in results_df.columns and pd.notna(results_df['label']).any():
            if 'xgb_prob' in results_df.columns and pd.notna(results_df['xgb_prob']).any():
                st.subheader("Model ROC / Performance")
                fpr, tpr, _ = roc_curve(results_df['label'], results_df['xgb_prob'])
                auc = roc_auc_score(results_df['label'], results_df['xgb_prob'])
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC (AUC = {auc:.3f})',
                    line=dict(color=PRIMARY_COLOR, width=2)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(dash='dash', color='gray')
                ))
                fig_roc.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=300
                )
                st.plotly_chart(fig_roc, use_container_width=True)
        
        # SHAP explanations
        if show_shap and xgb_model is not None and xgb_scaler is not None:
            st.subheader("SHAP Explanations")
            if not SHAP_AVAILABLE:
                st.warning("⚠️ SHAP is not installed. Install with: `pip install shap`")
                st.info("SHAP explanations are disabled. Other features still work!")
            else:
                try:
                    with st.spinner("Computing SHAP values (this may take a few seconds)..."):
                        top_k_indices = results_sorted.head(top_k).index.tolist()
                        shap_values, feature_names, summary_path, shap_values_full = compute_shap_and_save(
                            results_df, xgb_model, xgb_scaler, top_k_indices
                        )
                    
                    st.success("SHAP completed and saved to artifacts/")
                    
                    # Show summary image
                    if Path(summary_path).exists():
                        st.image(summary_path, use_container_width=True)
                    
                    # Show per-row explanations for top anomalies
                    st.write("**Top anomaly explanations:**")
                    for idx, (orig_idx, row) in enumerate(top_k_results.head(5).iterrows(), 1):
                        # Get SHAP values for this specific row
                        if orig_idx < len(shap_values_full):
                            shap_row = shap_values_full[orig_idx]
                        else:
                            shap_row = np.zeros(len(feature_names))
                        explanation = get_top_features_text(shap_row, feature_names)
                        st.write(f"{idx}. **User {row['user']}** ({row['date']}): {explanation}")
                
                except Exception as e:
                    st.error(f"SHAP computation failed: {e}")
                    with st.expander("Error details (expand for stack trace)"):
                        st.code(traceback.format_exc())
    
    # Footer status
    st.success("Saved results to artifacts/demo_scores.csv")

def main():
    # Page header
    st.markdown(f"<h1 style='text-align: center; color: {PRIMARY_COLOR}; font-size: 24px;'>Insider Threat Detection — Live Demo</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"<h2 style='color: {PRIMARY_COLOR}; font-size: 18px;'>Insider Threat Demo</h2>", unsafe_allow_html=True)
        
        # Mode selection
        mode = st.radio(
            "Mode",
            ["Use FastAPI", "Local models"],
            index=0 if st.session_state.use_api else 1
        )
        st.session_state.use_api = (mode == "Use FastAPI")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload activity CSV",
            type=['csv'],
            help="Upload a CSV file with columns: user, date, src_ip, dst_ip, file_path, success, label"
        )
        
        # Load sample button
        if st.button("Load sample"):
            sample_path = Path('data/sample_cert_small.csv')
            if sample_path.exists():
                uploaded_file = sample_path
                st.success("Sample data loaded!")
            else:
                st.error("Sample data not found. Please run: python scripts/create_dataset.py")
        
        # Top k slider
        top_k = st.slider("Top k anomalies", min_value=1, max_value=50, value=10, step=1)
        
        # SHAP checkbox (only show if SHAP is available)
        show_shap = st.checkbox("Show SHAP explanations", value=SHAP_AVAILABLE, disabled=not SHAP_AVAILABLE)
        if not SHAP_AVAILABLE:
            st.caption("⚠️ SHAP not installed. Install with: pip install shap")
        
        # Run inference button
        run_button = st.button("Run Inference", type="primary")
        
        # How to demo section
        with st.expander("How to demo"):
            st.markdown("""
            1. **Upload data**: Upload a CSV file or click "Load sample"
            2. **Run inference**: Click "Run Inference" button
            3. **Inspect results**: Review top anomalies, scores, and SHAP explanations
            """)
    
    # Main content area
    if run_button:
        if uploaded_file is None:
            st.error("Please upload a CSV file or load the sample data first.")
            return
        
        try:
            # Load data
            if isinstance(uploaded_file, Path) or isinstance(uploaded_file, str):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Show data preview
            st.subheader("Uploaded data preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Engineer features
            with st.spinner("Engineering features..."):
                features_df = engineer_features(df)
                st.session_state.features_df = features_df
            
            # Run inference
            if st.session_state.use_api:
                # Try FastAPI
                api_responses = call_api_batch(features_df)
                
                if api_responses is None:
                    st.warning("FastAPI not reachable — switching to local models (see docs/demo_instructions.md for help)")
                    st.session_state.use_api = False
                    st.session_state.api_reachable = False
                    
                    # Fall back to local models
                    xgb_model, xgb_scaler, iso_model, iso_scaler = load_local_models()
                    if xgb_model is None:
                        st.info("XGBoost model not found; XGBoost results and SHAP will be skipped.")
                    results_df = score_with_local_models(features_df, xgb_model, xgb_scaler, iso_model, iso_scaler)
                else:
                    # Process API responses
                    results_df = features_df.copy()
                    iso_scores = []
                    xgb_probs = []
                    xgb_preds = []
                    
                    for resp in api_responses:
                        if resp.get('iso_score') is not None:
                            iso_scores.append(-resp['iso_score'])  # Invert for consistency
                        else:
                            iso_scores.append(None)
                        
                        if resp.get('xgb_prediction') and 'probability_anomaly' in resp['xgb_prediction']:
                            xgb_probs.append(resp['xgb_prediction']['probability_anomaly'])
                            xgb_preds.append(resp['xgb_prediction']['prediction'])
                        else:
                            xgb_probs.append(None)
                            xgb_preds.append(None)
                    
                    results_df['iso_score'] = iso_scores
                    results_df['xgb_prob'] = xgb_probs
                    results_df['xgb_pred'] = xgb_preds
                    
                    # Load models for SHAP if needed
                    xgb_model, xgb_scaler, _, _ = load_local_models()
            else:
                # Use local models
                xgb_model, xgb_scaler, iso_model, iso_scaler = load_local_models()
                if xgb_model is None:
                    st.info("XGBoost model not found; XGBoost results and SHAP will be skipped.")
                results_df = score_with_local_models(features_df, xgb_model, xgb_scaler, iso_model, iso_scaler)
            
            st.session_state.results_df = results_df
            st.session_state.xgb_model = xgb_model
            st.session_state.xgb_scaler = xgb_scaler
            
            # Render results
            render_results(results_df, top_k, show_shap and xgb_model is not None, xgb_model, xgb_scaler)
        
        except Exception as e:
            st.error(f"Error processing data: {e}")
            with st.expander("Error details (expand for stack trace)"):
                st.code(traceback.format_exc())
    
    elif st.session_state.results_df is not None:
        # Show previous results if available
        st.info("Results from previous run. Click 'Run Inference' to process new data.")
        xgb_model = st.session_state.get('xgb_model')
        xgb_scaler = st.session_state.get('xgb_scaler')
        render_results(
            st.session_state.results_df,
            top_k,
            show_shap and xgb_model is not None,
            xgb_model,
            xgb_scaler
        )

if __name__ == "__main__":
    main()

