"""
Hyperparameter Grid Search Script

Purpose: Systematically test hyperparameter combinations to find optimal model settings.
This script runs a grid search over specified hyperparameter ranges and saves results
for comparison.

Usage:
    python scripts/hparam_search.py --model xgb --input data/features_train.csv --test_path data/features_test.csv
"""

import pandas as pd
import numpy as np
import argparse
import itertools
import json
from pathlib import Path
import time
import logging
from sklearn.metrics import roc_auc_score, f1_score
import sys

# Import training functions (simplified - would call actual training scripts)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def grid_search_xgb(X_train, y_train, X_test, y_test, param_grid):
    """
    Perform grid search for XGBoost.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Dictionary of hyperparameter ranges
    
    Returns:
        DataFrame with results for each combination
    """
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    
    results = []
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        
        logger.info(f"Testing: {params}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        start_time = time.time()
        model = xgb.XGBClassifier(**params, eval_metric='logloss', use_label_encoder=False)
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            **params,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'train_time': train_time
        })
        
        logger.info(f"  ROC-AUC: {roc_auc:.4f}, F1: {f1:.4f}, Time: {train_time:.2f}s")
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter grid search')
    parser.add_argument('--model', type=str, choices=['xgb', 'iso'], required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--output', type=str, default='artifacts/hparam_results.csv')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data...")
    train_df = pd.read_csv(args.input)
    test_df = pd.read_csv(args.test_path)
    
    # Prepare features (simplified - assumes standard format)
    feature_cols = [c for c in train_df.columns if c not in ['user', 'date', 'label']]
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    if args.model == 'xgb':
        # Define grid
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 500]
        }
        
        results_df = grid_search_xgb(X_train, y_train, X_test, y_test, param_grid)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    # Find best
    best = results_df.loc[results_df['roc_auc'].idxmax()]
    logger.info(f"\nBest parameters:")
    logger.info(f"  {best.to_dict()}")
    
    # Save best params
    best_path = output_path.parent / 'best_params.json'
    with open(best_path, 'w') as f:
        json.dump(best.to_dict(), f, indent=2)
    
    logger.info(f"✓ Results saved to: {output_path}")

if __name__ == '__main__':
    main()

