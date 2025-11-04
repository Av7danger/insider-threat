"""
Model Evaluation and Comparison Script

Purpose: Compare multiple models (XGBoost, Isolation Forest) on test data and generate
comprehensive evaluation reports with metrics and visualizations.

Metrics explained:
- Precision: Of all positive predictions, how many were correct? (Low false positives)
- Recall: Of all actual positives, how many did we catch? (Low false negatives)
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under ROC curve (measures overall classifier quality)
- Precision@k: For anomaly detection, precision in top k% most anomalous (useful when you can only investigate top alerts)

Why some metrics are preferred for anomaly detection:
- Precision@k: You can only investigate top alerts, so precision@1% tells you if top 1% are reliable
- Recall for rare class: Insider threats are rare, so we care about catching them (high recall)
- False positive cost: Too many false alarms waste analyst time, so precision matters too

How this file fits into the project:
- Runs after training models (train_xgb.py, train_iso.py)
- Produces comparison reports for model selection
- Generates visualizations for stakeholders

Usage:
    python scripts/evaluate.py --test_data data/features_test.csv
"""

import pandas as pd
import numpy as np
import argparse
import joblib
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_scaler(model_path, scaler_path):
    """Load a trained model and its scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def evaluate_xgb(model, scaler, X_test, y_test, feature_cols):
    """Evaluate XGBoost model."""
    X_test_scaled = scaler.transform(X_test[feature_cols].values)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba

def evaluate_iso(model, scaler, X_test, y_test, feature_cols):
    """Evaluate Isolation Forest model."""
    X_test_scaled = scaler.transform(X_test[feature_cols].values)
    
    # Get anomaly scores (lower = more anomalous)
    scores = model.score_samples(X_test_scaled)
    
    # Convert scores to predictions (assuming top 1% are anomalies)
    threshold = np.percentile(scores, 1)  # Bottom 1%
    y_pred = (scores < threshold).astype(int)
    
    # For metrics, we need labels
    if y_test is not None:
        metrics = {
            'precision_at_1pct': precision_score(y_test, y_pred, zero_division=0),
            'recall_at_1pct': recall_score(y_test, y_pred, zero_division=0),
            'f1_score_at_1pct': f1_score(y_test, y_pred, zero_division=0),
            'precision_at_5pct': None,  # Will calculate below
            'recall_at_5pct': None
        }
        
        # Top 5% precision
        threshold_5 = np.percentile(scores, 5)
        y_pred_5 = (scores < threshold_5).astype(int)
        metrics['precision_at_5pct'] = precision_score(y_test, y_pred_5, zero_division=0)
        metrics['recall_at_5pct'] = recall_score(y_test, y_pred_5, zero_division=0)
    else:
        metrics = {}
    
    # Convert scores to probabilities (invert and normalize)
    # Lower scores = higher anomaly probability
    scores_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min())
    
    return metrics, y_pred, scores_normalized, scores

def plot_roc_curves(results, output_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for model_name, result in results.items():
        if 'y_true' in result and 'y_proba' in result:
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_proba'])
            auc = roc_auc_score(result['y_true'], result['y_proba'])
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare models')
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test features CSV'
    )
    parser.add_argument(
        '--xgb_model',
        type=str,
        default='models/xgb_model.pkl',
        help='Path to XGBoost model (default: models/xgb_model.pkl)'
    )
    parser.add_argument(
        '--xgb_scaler',
        type=str,
        default='models/xgb_scaler.pkl',
        help='Path to XGBoost scaler (default: models/xgb_scaler.pkl)'
    )
    parser.add_argument(
        '--iso_model',
        type=str,
        default='models/iso_model.pkl',
        help='Path to Isolation Forest model (default: models/iso_model.pkl)'
    )
    parser.add_argument(
        '--iso_scaler',
        type=str,
        default='models/iso_scaler.pkl',
        help='Path to Isolation Forest scaler (default: models/iso_scaler.pkl)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='artifacts',
        help='Output directory for results (default: artifacts)'
    )
    
    args = parser.parse_args()
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_data}")
    test_df = pd.read_csv(args.test_data)
    logger.info(f"Test samples: {len(test_df):,}")
    
    # Check for labels
    has_labels = 'label' in test_df.columns
    
    # Prepare features
    exclude_cols = ['user', 'date', 'label', 'target', 'anomaly']
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]
    
    X_test = test_df[feature_cols]
    y_test = test_df['label'].values if has_labels else None
    
    if has_labels:
        logger.info(f"Label distribution: {np.bincount(y_test)}")
    
    results = {}
    all_metrics = []
    
    # Evaluate XGBoost if available
    if Path(args.xgb_model).exists() and Path(args.xgb_scaler).exists():
        logger.info("Evaluating XGBoost model...")
        try:
            model, scaler = load_model_and_scaler(args.xgb_model, args.xgb_scaler)
            metrics, y_pred, y_pred_proba = evaluate_xgb(model, scaler, test_df, y_test, feature_cols)
            
            results['XGBoost'] = {
                'metrics': metrics,
                'y_true': y_test,
                'y_proba': y_pred_proba if has_labels else None
            }
            
            all_metrics.append({
                'model': 'XGBoost',
                **metrics
            })
            
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        except Exception as e:
            logger.error(f"Error evaluating XGBoost: {e}")
    
    # Evaluate Isolation Forest if available
    if Path(args.iso_model).exists() and Path(args.iso_scaler).exists():
        logger.info("Evaluating Isolation Forest model...")
        try:
            model, scaler = load_model_and_scaler(args.iso_model, args.iso_scaler)
            metrics, y_pred, y_pred_proba, scores = evaluate_iso(model, scaler, test_df, y_test, feature_cols)
            
            results['Isolation Forest'] = {
                'metrics': metrics,
                'y_true': y_test,
                'y_proba': y_pred_proba if has_labels else None
            }
            
            all_metrics.append({
                'model': 'Isolation Forest',
                **metrics
            })
            
            if has_labels:
                logger.info(f"  Precision@1%: {metrics.get('precision_at_1pct', 0):.4f}")
                logger.info(f"  Recall@1%: {metrics.get('recall_at_1pct', 0):.4f}")
        except Exception as e:
            logger.error(f"Error evaluating Isolation Forest: {e}")
    
    # Save summary metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        summary_path = output_dir / 'summary_metrics.csv'
        metrics_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Saved summary metrics to: {summary_path}")
    
    # Generate comparison report
    report_path = output_dir / 'model_comparison.md'
    with open(report_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"Test Dataset: {args.test_data}\n")
        f.write(f"Test Samples: {len(test_df):,}\n\n")
        
        if has_labels:
            f.write(f"Label Distribution: {np.bincount(y_test)}\n\n")
        
        f.write("## Metrics Summary\n\n")
        if all_metrics:
            f.write(metrics_df.to_markdown(index=False))
            f.write("\n\n")
        
        for model_name, result in results.items():
            f.write(f"## {model_name}\n\n")
            metrics = result['metrics']
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    f.write(f"- **{metric_name}**: {metric_value:.4f}\n")
            f.write("\n")
    
    logger.info(f"✓ Saved comparison report to: {report_path}")
    
    # Plot ROC curves if labels available
    if has_labels and any('y_proba' in r and r['y_proba'] is not None for r in results.values()):
        roc_path = output_dir / 'roc.png'
        plot_roc_curves(results, roc_path)
        logger.info(f"✓ Saved ROC curves to: {roc_path}")
    
    logger.info("\nEvaluation complete!")

if __name__ == '__main__':
    main()

