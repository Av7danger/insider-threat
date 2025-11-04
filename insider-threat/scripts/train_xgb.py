"""
XGBoost Training Script

Purpose: Train a supervised classification model using XGBoost (gradient boosting).
This model requires labeled data (normal vs. anomalous) and learns to distinguish
between them by combining many weak decision trees into a strong classifier.

XGBoost in plain words:
- Builds decision trees one at a time, each trying to correct the previous tree's mistakes
- Uses gradient descent to find the best splits
- Combines predictions from all trees (ensemble method)
- Very effective for tabular data with mixed feature types

Class imbalance handling:
- Insider threats are rare (maybe 1% of data is anomalous)
- Models tend to ignore rare classes and predict "normal" all the time
- Solutions:
  * class_weight: penalize misclassifying rare class more
  * sample_weight: give more weight to rare class examples
  * upsampling: duplicate rare examples
  * downsampling: remove some normal examples
  * This script uses class_weight='balanced' as a simple fix

How this file fits into the project:
- Requires labels (runs after data_prep.py with labeled data)
- Outputs a model that can provide probability scores for predictions
- Used for comparison with unsupervised models and for explainability (SHAP)

Usage:
    python scripts/train_xgb.py --input data/features_train.csv --test_path data/features_test.csv
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_features_and_labels(df, label_col='label'):
    """
    Extract features and labels from dataframe.
    
    Args:
        df: dataframe with features and labels
        label_col: name of label column
    
    Returns:
        X: feature matrix
        y: label array
        feature_names: list of feature column names
    """
    # Check if label column exists
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available columns: {df.columns.tolist()}")
    
    # Exclude identifiers and labels from features
    exclude_cols = ['user', 'date', 'label', 'target', 'anomaly']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[label_col].values
    
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    logger.info(f"Label distribution: {np.bincount(y.astype(int))}")
    
    return X, y, feature_cols

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost classifier')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to training features CSV file'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        required=True,
        help='Path to test features CSV file'
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default='models/xgb_model.pkl',
        help='Path to save trained model (default: models/xgb_model.pkl)'
    )
    parser.add_argument(
        '--output_scaler',
        type=str,
        default='models/xgb_scaler.pkl',
        help='Path to save scaler (default: models/xgb_scaler.pkl)'
    )
    parser.add_argument(
        '--output_metrics',
        type=str,
        default='artifacts/xgb_metrics.json',
        help='Path to save evaluation metrics (default: artifacts/xgb_metrics.json)'
    )
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of boosting rounds (default: 100)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=5,
        help='Maximum tree depth (default: 5)'
    )
    parser.add_argument(
        '--early_stopping_rounds',
        type=int,
        default=10,
        help='Early stopping rounds (default: 10)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--label_col',
        type=str,
        default='label',
        help='Name of label column (default: label)'
    )
    
    args = parser.parse_args()
    
    # Load training data
    logger.info(f"Loading training data from: {args.input}")
    train_df = pd.read_csv(args.input)
    logger.info(f"Training samples: {len(train_df):,}")
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_path}")
    test_df = pd.read_csv(args.test_path)
    logger.info(f"Test samples: {len(test_df):,}")
    
    # Prepare features and labels
    X_train, y_train, feature_names = prepare_features_and_labels(train_df, args.label_col)
    X_test, y_test, _ = prepare_features_and_labels(test_df, args.label_col)
    
    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Check class imbalance
    class_counts = np.bincount(y_train.astype(int))
    imbalance_ratio = class_counts[1] / class_counts[0] if len(class_counts) > 1 and class_counts[0] > 0 else 0
    logger.info(f"Class imbalance ratio (positive/negative): {imbalance_ratio:.4f}")
    
    if imbalance_ratio < 0.1:
        logger.warning("Severe class imbalance detected. Using balanced class weights.")
        use_balanced = True
    else:
        use_balanced = False
    
    # Train XGBoost
    logger.info("Training XGBoost classifier...")
    logger.info(f"  n_estimators: {args.n_estimators}")
    logger.info(f"  learning_rate: {args.learning_rate}")
    logger.info(f"  max_depth: {args.max_depth}")
    
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.random_state,
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=(1.0 / imbalance_ratio) if use_balanced and imbalance_ratio > 0 else 1.0
    )
    
    # Use early stopping with validation set
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # Newer XGBoost versions use different API
    try:
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_split, y_val_split)],
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=False
        )
    except TypeError:
        # Fallback for newer XGBoost versions
        model.fit(X_train_scaled, y_train)
    
    logger.info("✓ Model training complete")
    
    # Make predictions
    logger.info("Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'precision': float(report['1']['precision']) if '1' in report else 0.0,
        'recall': float(report['1']['recall']) if '1' in report else 0.0,
        'f1_score': float(report['1']['f1-score']) if '1' in report else 0.0,
        'classification_report': report
    }
    
    logger.info(f"\nTest Set Metrics:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # Save model and scaler
    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model_path)
    logger.info(f"✓ Saved model to: {output_model_path}")
    
    output_scaler_path = Path(args.output_scaler)
    output_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_scaler_path)
    logger.info(f"✓ Saved scaler to: {output_scaler_path}")
    
    # Save metrics
    output_metrics_path = Path(args.output_metrics)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Saved metrics to: {output_metrics_path}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = output_metrics_path.parent / 'xgb_confusion_matrix.png'
    plt.savefig(cm_path)
    logger.info(f"✓ Saved confusion matrix to: {cm_path}")
    plt.close()
    
    logger.info("\nNext steps:")
    logger.info("  1. Run evaluation: python scripts/evaluate.py --test_data data/features_test.csv")
    logger.info("  2. Generate SHAP explanations: python scripts/explain_xgb_shap.py")
    logger.info("  3. Compare with Isolation Forest model")

if __name__ == '__main__':
    main()

