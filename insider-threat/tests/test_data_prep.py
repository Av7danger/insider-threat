"""
Unit Tests for Data Preparation

Purpose: Test that data_prep.py correctly aggregates features and handles edge cases.

How to run:
    pytest tests/test_data_prep.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from data_prep import engineer_features, handle_missing_values, find_timestamp_column

def test_engineer_features_basic():
    """Test basic feature engineering."""
    # Create synthetic event data
    df = pd.DataFrame({
        'user': ['user1', 'user1', 'user2'],
        'date': ['2020-01-01', '2020-01-01', '2020-01-01'],
        'src_ip': ['1.1.1.1', '1.1.1.2', '2.2.2.2'],
        'file_path': ['/file1', '/file2', '/file3'],
        'success': [1, 1, 0]
    })
    
    df['date'] = pd.to_datetime(df['date'])
    
    features = engineer_features(df, 'date', 'user', None)
    
    # Check that features were created
    assert len(features) > 0
    assert 'total_events' in features.columns
    assert 'user' in features.columns

def test_handle_missing_values():
    """Test missing value handling."""
    df = pd.DataFrame({
        'total_events': [10, None, 30],
        'avg_success': [0.9, 0.8, None],
        'start_hour': [9, 10, None]
    })
    
    df_cleaned = handle_missing_values(df)
    
    # Check that missing values were filled
    assert df_cleaned['total_events'].isna().sum() == 0
    assert df_cleaned['avg_success'].isna().sum() == 0
    assert df_cleaned['start_hour'].isna().sum() == 0

def test_find_timestamp_column():
    """Test timestamp column detection."""
    df = pd.DataFrame({
        'timestamp': ['2020-01-01', '2020-01-02'],
        'other_col': [1, 2]
    })
    
    timestamp_col = find_timestamp_column(df)
    assert timestamp_col == 'timestamp'

def test_feature_aggregation_counts():
    """Test that aggregation counts are correct."""
    # Create data with known counts
    df = pd.DataFrame({
        'user': ['user1'] * 5,  # 5 events for user1
        'date': pd.to_datetime(['2020-01-01'] * 5),
        'src_ip': ['1.1.1.1', '1.1.1.2', '1.1.1.1', '1.1.1.3', '1.1.1.1'],
        'file_path': ['/file1', '/file2', '/file1', '/file3', '/file1'],
        'success': [1, 1, 1, 0, 1]
    })
    
    features = engineer_features(df, 'date', 'user', None)
    
    # Check aggregation
    user1_features = features[features['user'] == 'user1'].iloc[0]
    assert user1_features['total_events'] == 5
    assert user1_features['unique_src_ip'] == 3  # 3 unique IPs
    assert user1_features['distinct_files'] == 3  # 3 unique files

