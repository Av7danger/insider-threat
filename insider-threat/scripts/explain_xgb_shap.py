"""
SHAP Explainability Script for XGBoost

Purpose: Generate SHAP (SHapley Additive exPlanations) values to explain XGBoost predictions.
SHAP values show how much each feature contributes to each prediction, helping security
analysts understand why a user was flagged as anomalous.

What SHAP values mean:
- Positive SHAP value: feature pushes prediction toward "anomalous"
- Negative SHAP value: feature pushes prediction toward "normal"
- Large absolute value: feature has strong influence
- Each prediction's SHAP values sum to (prediction - average prediction)

How to read summary plot:
- Y-axis: features sorted by importance
- X-axis: SHAP value (impact on prediction)
- Color: feature value (red = high, blue = low)
- Each dot is one prediction
- Pattern: if red dots are on right side, high values of that feature increase anomaly score

How explainability helps triage alerts:
- Instead of just "user X is anomalous", you get "user X is anomalous because they accessed 50 files (normal: 5 files)"
- Helps prioritize: flag users with unusual file access patterns vs. users with unusual IPs
- Builds trust: analysts understand why model flagged someone

How this file fits into the project:
- Runs after training XGBoost (train_xgb.py)
- Generates visualizations for security analysts
- Can be integrated into API for real-time explanations

Usage:
    python scripts/explain_xgb_shap.py --model_path models/xgb_model.pkl --test_data data/features_test.csv
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import shap
from pathlib import Path
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate SHAP explanations for XGBoost')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained XGBoost model'
    )
    parser.add_argument(
        '--scaler_path',
        type=str,
        default='models/xgb_scaler.pkl',
        help='Path to scaler (default: models/xgb_scaler.pkl)'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test features CSV'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=200,
        help='Number of samples to explain (default: 200, more = slower)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='artifacts',
        help='Output directory (default: artifacts)'
    )
    
    args = parser.parse_args()
    
    # Load model and scaler
    logger.info(f"Loading model from: {args.model_path}")
    model = joblib.load(args.model_path)
    scaler = joblib.load(args.scaler_path)
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    test_df = pd.read_csv(args.test_data)
    
    # Prepare features
    exclude_cols = ['user', 'date', 'label', 'target', 'anomaly']
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]
    
    X_test = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    # Sample subset for faster computation
    if len(X_test_scaled) > args.n_samples:
        indices = np.random.choice(len(X_test_scaled), args.n_samples, replace=False)
        X_test_sample = X_test_scaled[indices]
        test_df_sample = test_df.iloc[indices]
    else:
        X_test_sample = X_test_scaled
        test_df_sample = test_df
    
    logger.info(f"Computing SHAP values for {len(X_test_sample)} samples...")
    logger.info("This may take a few minutes...")
    
    # Create SHAP explainer
    # TreeExplainer is fast for tree-based models like XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_sample)
    
    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class SHAP values
    
    logger.info("✓ SHAP values computed")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary plot
    logger.info("Generating summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols, show=False)
    summary_path = output_dir / 'shap_summary.png'
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"✓ Saved summary plot to: {summary_path}")
    
    # Waterfall plots for top anomalies (if labels exist)
    if 'label' in test_df_sample.columns:
        # Find top anomalous predictions
        predictions = model.predict_proba(X_test_sample)[:, 1]
        anomaly_indices = np.argsort(predictions)[-5:][::-1]  # Top 5 highest probability
        
        logger.info("Generating force plots for top 5 anomalies...")
        for i, idx in enumerate(anomaly_indices):
            user = test_df_sample.iloc[idx]['user'] if 'user' in test_df_sample.columns else f'sample_{idx}'
            
            # Create force plot
            shap.force_plot(
                explainer.expected_value,
                shap_values[idx],
                X_test_sample[idx],
                feature_names=feature_cols,
                matplotlib=True,
                show=False
            )
            
            force_path = output_dir / f'shap_force_anomaly_{i+1}_{user}.png'
            plt.savefig(force_path, bbox_inches='tight', dpi=150)
            plt.close()
            logger.info(f"  Saved force plot {i+1} to: {force_path}")
    
    # Feature importance plot (alternative visualization)
    logger.info("Generating feature importance plot...")
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols, plot_type="bar", show=False)
    importance_path = output_dir / 'shap_importance.png'
    plt.savefig(importance_path, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f"✓ Saved importance plot to: {importance_path}")
    
    logger.info("\nSHAP explanation complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Review shap_summary.png to understand global feature importance")
    logger.info("  2. Check force plots for individual anomaly explanations")
    logger.info("  3. Integrate SHAP into API for real-time explanations")

if __name__ == '__main__':
    main()

