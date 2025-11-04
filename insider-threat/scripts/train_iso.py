"""
Isolation Forest Training Script

Purpose: Train an unsupervised anomaly detection model using Isolation Forest.
This model doesn't require labels - it learns what "normal" looks like and flags
anything that deviates significantly from normal patterns.

What Isolation Forest does (in plain words):
- Imagine you have a forest of decision trees
- Each tree randomly picks features and splits on them
- Normal data points are easy to isolate (few splits needed)
- Anomalous data points are hard to isolate (many splits needed)
- The "isolation score" measures how many splits it takes to isolate a point
- Low scores = anomalies (they're different from normal patterns)

Why scaling matters:
- Features have different scales (e.g., total_events might be 0-1000, while avg_success is 0-1)
- Without scaling, features with larger ranges dominate the model
- StandardScaler centers each feature at 0 and scales to unit variance
- This ensures all features contribute equally to anomaly detection

How this file fits into the project:
- Runs after data_prep.py (uses features.csv)
- Outputs a model that can detect anomalies without labels
- Used for comparison with supervised models (XGBoost)

Usage:
    python scripts/train_iso.py --input data/features_train.csv --contamination 0.01
"""

import pandas as pd
import numpy as np
import argparse
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_features(df):
    """
    Extract and prepare feature columns for training.
    
    We exclude:
    - user, date (identifiers, not features)
    - label (not used in unsupervised learning)
    """
    # Identify feature columns (exclude identifiers and labels)
    exclude_cols = ['user', 'date', 'label', 'target', 'anomaly']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    
    X = df[feature_cols].values
    return X, feature_cols

def flag_anomalies(scores_csv_path, threshold=None):
    """
    Helper function to print top anomalies with context.
    
    This helps security analysts triage alerts by showing:
    - Which users/dates are flagged
    - Their anomaly scores (how anomalous they are)
    - Context to investigate further
    
    Args:
        scores_csv_path: path to CSV with columns: user, date, anomaly_score
        threshold: optional threshold to filter anomalies (default: use top 1%)
    """
    scores_df = pd.read_csv(scores_csv_path)
    
    if threshold is None:
        # Use top 1% as default
        threshold = scores_df['anomaly_score'].quantile(0.99)
    
    anomalies = scores_df[scores_df['anomaly_score'] < threshold].sort_values('anomaly_score')
    
    print(f"\n{'='*80}")
    print(f"TOP ANOMALIES (threshold: {threshold:.4f})")
    print(f"{'='*80}")
    print(f"Total anomalies found: {len(anomalies)}")
    print(f"\nTop 20 anomalies:")
    print("-"*80)
    
    for idx, row in anomalies.head(20).iterrows():
        print(f"User: {row['user']}, Date: {row['date']}, Score: {row['anomaly_score']:.4f}")
    
    return anomalies

def main():
    parser = argparse.ArgumentParser(description='Train Isolation Forest anomaly detector')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input features CSV file'
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default='models/iso_model.pkl',
        help='Path to save trained model (default: models/iso_model.pkl)'
    )
    parser.add_argument(
        '--output_scaler',
        type=str,
        default='models/iso_scaler.pkl',
        help='Path to save scaler (default: models/iso_scaler.pkl)'
    )
    parser.add_argument(
        '--output_scores',
        type=str,
        default='models/iso_train_scores.csv',
        help='Path to save anomaly scores (default: models/iso_train_scores.csv)'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.01,
        help='Expected proportion of anomalies (default: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of trees in the forest (default: 100)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default=None,
        help='Optional path to test data for evaluation'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading features from: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} rows")
    
    # Prepare features
    X, feature_cols = prepare_features(df)
    logger.info(f"Feature matrix shape: {X.shape}")
    
    # Scale features (critical for Isolation Forest)
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    logger.info(f"Training Isolation Forest...")
    logger.info(f"  Contamination: {args.contamination} (expecting {args.contamination*100:.1f}% anomalies)")
    logger.info(f"  Number of trees: {args.n_estimators}")
    
    model = IsolationForest(
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_scaled)
    logger.info("✓ Model training complete")
    
    # Compute anomaly scores for training data
    logger.info("Computing anomaly scores...")
    # Note: decision_function returns negative scores (lower = more anomalous)
    # We'll use score_samples to get the actual isolation score
    scores = model.score_samples(X_scaled)
    predictions = model.predict(X_scaled)  # -1 for anomaly, 1 for normal
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'user': df['user'].values if 'user' in df.columns else range(len(df)),
        'date': df['date'].values if 'date' in df.columns else ['unknown'] * len(df),
        'anomaly_score': scores,
        'prediction': predictions
    })
    
    # Save model and scaler
    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model_path)
    logger.info(f"✓ Saved model to: {output_model_path}")
    
    output_scaler_path = Path(args.output_scaler)
    output_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_scaler_path)
    logger.info(f"✓ Saved scaler to: {output_scaler_path}")
    
    # Save scores
    output_scores_path = Path(args.output_scores)
    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_scores_path, index=False)
    logger.info(f"✓ Saved scores to: {output_scores_path}")
    
    # Print summary
    n_anomalies = (predictions == -1).sum()
    logger.info(f"\nTraining Summary:")
    logger.info(f"  Total samples: {len(df):,}")
    logger.info(f"  Anomalies detected: {n_anomalies:,} ({n_anomalies/len(df)*100:.2f}%)")
    logger.info(f"  Expected anomalies: {int(args.contamination * len(df)):,}")
    
    # Evaluate on test data if provided
    if args.test_data:
        logger.info(f"\nEvaluating on test data: {args.test_data}")
        test_df = pd.read_csv(args.test_data)
        X_test, _ = prepare_features(test_df)
        X_test_scaled = scaler.transform(X_test)
        
        test_scores = model.score_samples(X_test_scaled)
        test_predictions = model.predict(X_test_scaled)
        
        test_results_df = pd.DataFrame({
            'user': test_df['user'].values if 'user' in test_df.columns else range(len(test_df)),
            'date': test_df['date'].values if 'date' in test_df.columns else ['unknown'] * len(test_df),
            'anomaly_score': test_scores,
            'prediction': test_predictions
        })
        
        test_output_path = Path(args.output_scores).parent / 'iso_test_scores.csv'
        test_results_df.to_csv(test_output_path, index=False)
        logger.info(f"✓ Saved test scores to: {test_output_path}")
        
        n_test_anomalies = (test_predictions == -1).sum()
        logger.info(f"  Test anomalies detected: {n_test_anomalies:,} ({n_test_anomalies/len(test_df)*100:.2f}%)")
    
    logger.info("\nNext steps:")
    logger.info("  1. Review anomalies: python -c 'from scripts.train_iso import flag_anomalies; flag_anomalies(\"models/iso_train_scores.csv\")'")
    logger.info("  2. Run evaluation: python scripts/evaluate.py --test_data data/features_test.csv")
    logger.info("  3. Compare with supervised models (XGBoost) if labels are available")

if __name__ == '__main__':
    main()

