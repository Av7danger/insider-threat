"""
Data Preparation and Feature Engineering Script

Purpose: This script aggregates raw user activity data into per-user-per-day features
that are suitable for machine learning models. It converts time-series event data
into a tabular format with behavioral features.

If you're new to this:
- Raw event logs have many rows per user (one row per event)
- We aggregate events by user and day to create "behavioral snapshots"
- Each feature captures a different aspect of user behavior:
  * total_events: how active the user was
  * unique_src_ip/dst_ip: how many different locations they accessed
  * distinct_files: how many different files they touched
  * avg_success: whether their actions succeeded or failed
  * start_hour/end_hour/peak_hour: when they were active (helps detect off-hours activity)

How this file fits into the project:
- Runs after schema detection (schema_and_inventory.py)
- Outputs features that feed into all training scripts (train_iso.py, train_xgb.py, train_lstm.py)
- Creates train/test splits for supervised learning

Usage:
    python scripts/data_prep.py --input data/cert_dataset.csv --output data/features.csv
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

# Setup logging for beginners to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_timestamp_column(df):
    """
    Automatically detect which column contains timestamps.
    
    We look for columns that:
    - Have 'time', 'date', 'timestamp' in the name, OR
    - Can be parsed as datetime
    
    Returns:
        str: name of timestamp column, or None if not found
    """
    # Try common timestamp column names
    timestamp_keywords = ['time', 'date', 'timestamp', 'datetime']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in timestamp_keywords):
            try:
                pd.to_datetime(df[col].dropna().head(100))
                logger.info(f"Found timestamp column: {col}")
                return col
            except:
                continue
    
    # Try to auto-detect by parsing all columns
    for col in df.columns:
        try:
            pd.to_datetime(df[col].dropna().head(100))
            logger.info(f"Auto-detected timestamp column: {col}")
            return col
        except:
            continue
    
    return None

def find_user_column(df):
    """
    Find the column that identifies users.
    
    Looks for columns with 'user', 'id', 'username' in the name.
    """
    user_keywords = ['user', 'id', 'username', 'employee']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in user_keywords):
            return col
    
    # If no obvious user column, try first column
    logger.warning(f"No user column found, using first column: {df.columns[0]}")
    return df.columns[0]

def find_label_column(df):
    """
    Find the column that contains labels (0/1 for normal/anomaly).
    
    Returns None if no label column exists (unsupervised learning scenario).
    """
    label_keywords = ['label', 'target', 'anomaly', 'malicious']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in label_keywords):
            return col
    return None

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
        # If values are large (> 1e12), might be milliseconds
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
    
    raise ValueError(f"Could not parse timestamps from series: {series.head()}")

def engineer_features(df, timestamp_col, user_col, label_col=None):
    """
    Aggregate events into per-user-per-day features.
    
    This is the core feature engineering function. It groups events by user and day,
    then computes statistical features that capture behavioral patterns.
    
    Args:
        df: raw event dataframe
        timestamp_col: column name for timestamps
        user_col: column name for user identifiers
        label_col: optional column name for labels
    
    Returns:
        DataFrame with columns: user, date, and feature columns
    """
    logger.info("Parsing timestamps...")
    df['_parsed_timestamp'] = parse_timestamp_robust(df[timestamp_col])
    
    # Extract date (ignore time for daily aggregation)
    df['_date'] = df['_parsed_timestamp'].dt.date
    
    # Identify numeric columns (for aggregation)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if timestamp_col in numeric_cols:
        numeric_cols.remove(timestamp_col)
    
    # Identify categorical/string columns (for counting unique values)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    logger.info(f"Aggregating features by user and date...")
    logger.info(f"Unique users: {df[user_col].nunique()}")
    logger.info(f"Date range: {df['_date'].min()} to {df['_date'].max()}")
    
    # Group by user and date
    grouped = df.groupby([user_col, '_date'])
    
    # Initialize feature list
    features_list = []
    
    # Process each user-day group
    for (user, date), group in grouped:
        feature_row = {
            'user': user,
            'date': date
        }
        
        # Total number of events (activity volume)
        feature_row['total_events'] = len(group)
        
        # Unique source IPs (geographic diversity)
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
        
        # Distinct files accessed
        file_cols = [c for c in group.columns if 'file' in c.lower() or 'path' in c.lower()]
        if file_cols:
            feature_row['distinct_files'] = group[file_cols[0]].nunique()
        else:
            feature_row['distinct_files'] = 0
        
        # Average success rate (if success/failure column exists)
        success_cols = [c for c in group.columns if 'success' in c.lower() or 'status' in c.lower()]
        if success_cols:
            # Convert to binary (1 for success, 0 for failure)
            success_values = group[success_cols[0]].astype(str).str.lower()
            success_binary = success_values.isin(['1', 'true', 'success', 'succeeded', 'ok', '200']).astype(int)
            feature_row['avg_success'] = success_binary.mean()
        else:
            feature_row['avg_success'] = 1.0  # Assume success if no column
        
        # Time-based features (when did user start/end activity)
        hours = group['_parsed_timestamp'].dt.hour
        feature_row['start_hour'] = hours.min()
        feature_row['end_hour'] = hours.max()
        feature_row['peak_hour'] = hours.mode().iloc[0] if len(hours.mode()) > 0 else hours.median()
        
        # Add label if present (use max - if any event in user-day is anomalous, mark as anomalous)
        if label_col:
            feature_row['label'] = int(group[label_col].max())
        
        features_list.append(feature_row)
    
    features_df = pd.DataFrame(features_list)
    logger.info(f"Generated {len(features_df)} user-day feature rows")
    
    return features_df

def handle_missing_values(df):
    """
    Fill missing values with sensible defaults.
    
    Why we fill with 0/median:
    - 0 for counts (no activity = 0 events)
    - Median for continuous features (preserves distribution)
    - This prevents ML models from failing on missing values
    """
    logger.info("Handling missing values...")
    
    # Count columns: fill with 0 (no activity)
    count_cols = ['total_events', 'unique_src_ip', 'unique_dst_ip', 'distinct_files']
    for col in count_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                df[col].fillna(0, inplace=True)
                logger.info(f"  Filled {missing} missing values in {col} with 0")
    
    # Continuous columns: fill with median
    continuous_cols = ['avg_success', 'start_hour', 'end_hour', 'peak_hour']
    for col in continuous_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  Filled {missing} missing values in {col} with median ({median_val})")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Prepare features from raw event data'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (raw event data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/features.csv',
        help='Path to output features CSV (default: data/features.csv)'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Also create train/test splits (requires label column)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test set size for train/test split (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Check input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    logger.info(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Detect columns
    timestamp_col = find_timestamp_column(df)
    if not timestamp_col:
        logger.error("Could not find timestamp column. Please specify manually.")
        sys.exit(1)
    
    user_col = find_user_column(df)
    label_col = find_label_column(df)
    
    logger.info(f"Using columns:")
    logger.info(f"  Timestamp: {timestamp_col}")
    logger.info(f"  User: {user_col}")
    if label_col:
        logger.info(f"  Label: {label_col}")
    else:
        logger.info(f"  Label: None (unsupervised learning)")
    
    # Engineer features
    features_df = engineer_features(df, timestamp_col, user_col, label_col)
    
    # Handle missing values
    features_df = handle_missing_values(features_df)
    
    # Save features
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved features to: {output_path}")
    logger.info(f"  Features shape: {features_df.shape}")
    
    # Create train/test splits if requested and label exists
    if args.split and label_col:
        logger.info("Creating train/test splits...")
        train_df, test_df = train_test_split(
            features_df,
            test_size=args.test_size,
            stratify=features_df['label'],
            random_state=42
        )
        
        train_path = output_path.parent / 'features_train.csv'
        test_path = output_path.parent / 'features_test.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"✓ Saved train split: {train_path} ({len(train_df)} rows)")
        logger.info(f"✓ Saved test split: {test_path} ({len(test_df)} rows)")
    elif args.split and not label_col:
        logger.warning("Cannot create train/test split: no label column found")
    
    logger.info("\nFeature engineering complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Train Isolation Forest: python scripts/train_iso.py --input data/features.csv")
    if label_col:
        logger.info("  2. Train XGBoost: python scripts/train_xgb.py --input data/features_train.csv")

if __name__ == '__main__':
    main()

