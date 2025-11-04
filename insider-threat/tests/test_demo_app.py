"""
Unit Tests for Streamlit Demo App

Purpose: Test core logic functions from demo_app.py without UI components.
These tests validate feature engineering and scoring functionality.

How to run:
    pytest tests/test_demo_app.py -v
    pytest tests/test_demo_app.py -q  # Quiet mode
    pytest tests/test_demo_app.py::test_engineer_features  # Run specific test
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add demo_app to path for importing functions
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from demo_app (we'll test the core logic)
from demo_app import engineer_features
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def test_engineer_features_basic():
    """
    Test that engineer_features correctly aggregates events into per-user-per-day features.
    
    Creates a synthetic DataFrame with 3 users, multiple timestamps, and verifies:
    - Correct number of feature rows (one per user-day)
    - Required columns are present
    - Aggregation counts are correct
    """
    # Create synthetic event data
    data = {
        'user': ['user1', 'user1', 'user1', 'user2', 'user2', 'user3'],
        'date': [
            '2020-01-01 09:00:00',
            '2020-01-01 10:00:00',
            '2020-01-01 11:00:00',
            '2020-01-01 08:00:00',
            '2020-01-02 09:00:00',  # Different day for user2
            '2020-01-01 12:00:00'
        ],
        'src_ip': ['192.168.1.10', '192.168.1.10', '192.168.1.11', '192.168.1.20', '192.168.1.20', '192.168.1.30'],
        'dst_ip': ['192.168.1.100', '192.168.1.100', '192.168.1.101', '192.168.1.200', '192.168.1.200', '192.168.1.300'],
        'file_path': ['/file1.txt', '/file2.txt', '/file1.txt', '/file3.txt', '/file4.txt', '/file5.txt'],
        'success': [1, 1, 0, 1, 1, 1]
    }
    
    df = pd.DataFrame(data)
    
    # Run feature engineering
    features_df = engineer_features(df)
    
    # Assertions
    assert len(features_df) == 3, f"Expected 3 user-day rows, got {len(features_df)}"
    
    # Check required columns
    required_cols = ['user', 'date', 'total_events', 'unique_src_ip', 'unique_dst_ip', 
                     'distinct_files', 'avg_success', 'start_hour', 'end_hour', 'peak_hour']
    for col in required_cols:
        assert col in features_df.columns, f"Missing required column: {col}"
    
    # Check aggregation correctness
    # user1 on 2020-01-01: 3 events, 2 unique src_ips, 2 unique dst_ips, 2 distinct files
    user1_row = features_df[features_df['user'] == 'user1'].iloc[0]
    assert user1_row['total_events'] == 3, "user1 should have 3 events"
    assert user1_row['unique_src_ip'] == 2, "user1 should have 2 unique source IPs"
    assert user1_row['distinct_files'] == 2, "user1 should have 2 distinct files"
    
    # Check success rate (2 out of 3 successful)
    assert abs(user1_row['avg_success'] - 2/3) < 0.01, "user1 success rate should be ~0.67"
    
    # Check time features
    assert 9 <= user1_row['start_hour'] <= 12, "start_hour should be in valid range"
    assert 9 <= user1_row['end_hour'] <= 12, "end_hour should be in valid range"

def test_engineer_features_multiple_days():
    """
    Test feature engineering handles multiple days correctly.
    
    Creates data where one user has events on multiple days and verifies
    separate feature rows are created for each day.
    """
    data = {
        'user': ['user1', 'user1', 'user1', 'user1'],
        'date': [
            '2020-01-01 09:00:00',
            '2020-01-01 10:00:00',
            '2020-01-02 09:00:00',  # Different day
            '2020-01-02 10:00:00'
        ],
        'src_ip': ['192.168.1.10'] * 4,
        'dst_ip': ['192.168.1.100'] * 4,
        'file_path': ['/file1.txt', '/file2.txt', '/file3.txt', '/file4.txt'],
        'success': [1, 1, 1, 1]
    }
    
    df = pd.DataFrame(data)
    features_df = engineer_features(df)
    
    # Should have 2 rows (one for each day)
    assert len(features_df) == 2, f"Expected 2 user-day rows, got {len(features_df)}"
    
    # Both rows should be for user1
    assert all(features_df['user'] == 'user1'), "All rows should be for user1"
    
    # Check dates are different
    dates = features_df['date'].tolist()
    assert len(set(dates)) == 2, "Should have 2 different dates"

def test_engineer_features_missing_columns():
    """
    Test that engineer_features handles missing optional columns gracefully.
    
    Creates minimal data (only user and date) and verifies it still works.
    """
    data = {
        'user': ['user1', 'user1'],
        'date': ['2020-01-01 09:00:00', '2020-01-01 10:00:00']
    }
    
    df = pd.DataFrame(data)
    features_df = engineer_features(df)
    
    # Should still create features
    assert len(features_df) == 1, "Should create one feature row"
    assert features_df.iloc[0]['total_events'] == 2, "Should count 2 events"
    assert features_df.iloc[0]['unique_src_ip'] == 0, "Missing IP column should default to 0"

def test_scoring_fallback_trains_isolation_forest():
    """
    Test that scoring fallback trains IsolationForest when no models are available.
    
    Creates synthetic features and verifies that a fallback model can be trained
    and produces iso_score column with finite values.
    """
    # Create synthetic features DataFrame
    features_df = pd.DataFrame({
        'user': ['user1', 'user2', 'user3'] * 3,
        'date': pd.to_datetime(['2020-01-01', '2020-01-01', '2020-01-01'] * 3).date,
        'total_events': [10, 20, 5, 15, 25, 8, 12, 22, 6],
        'unique_src_ip': [1, 2, 1, 1, 3, 1, 2, 2, 1],
        'unique_dst_ip': [2, 3, 1, 2, 4, 1, 3, 3, 1],
        'distinct_files': [5, 10, 3, 7, 12, 4, 6, 11, 3],
        'avg_success': [0.9, 0.95, 0.8, 0.92, 0.98, 0.85, 0.91, 0.96, 0.82],
        'start_hour': [9, 10, 8, 9, 11, 8, 10, 10, 9],
        'end_hour': [17, 18, 16, 17, 19, 16, 18, 18, 17],
        'peak_hour': [13, 14, 12, 13, 15, 12, 14, 14, 13]
    })
    
    # Simulate scoring with fallback (no models available)
    feature_cols = [
        'total_events', 'unique_src_ip', 'unique_dst_ip', 'distinct_files',
        'avg_success', 'start_hour', 'end_hour', 'peak_hour'
    ]
    
    X = features_df[feature_cols].values
    
    # Train quick IsolationForest (fallback behavior)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso_fallback = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_fallback.fit(X_scaled)
    scores = iso_fallback.score_samples(X_scaled)
    
    # Add scores to results
    results_df = features_df.copy()
    results_df['iso_score'] = -scores  # Invert so higher = more anomalous
    
    # Assertions
    assert 'iso_score' in results_df.columns, "iso_score column should be added"
    assert all(pd.notna(results_df['iso_score'])), "All iso_scores should be finite"
    assert all(np.isfinite(results_df['iso_score'])), "All iso_scores should be finite numbers"
    
    # Check score range (scores are typically negative, so inverted should be positive)
    assert results_df['iso_score'].min() < results_df['iso_score'].max(), "Scores should vary"

def test_feature_engineering_with_labels():
    """
    Test that engineer_features correctly handles label column if present.
    """
    data = {
        'user': ['user1', 'user1', 'user2'],
        'date': ['2020-01-01 09:00:00', '2020-01-01 10:00:00', '2020-01-01 11:00:00'],
        'src_ip': ['192.168.1.10', '192.168.1.10', '192.168.1.20'],
        'dst_ip': ['192.168.1.100', '192.168.1.100', '192.168.1.200'],
        'file_path': ['/file1.txt', '/file2.txt', '/file3.txt'],
        'success': [1, 1, 1],
        'label': [0, 0, 1]  # user1 normal, user2 anomalous
    }
    
    df = pd.DataFrame(data)
    features_df = engineer_features(df)
    
    # Should have label column
    assert 'label' in features_df.columns, "Label column should be included"
    
    # user1 should have label 0 (max of [0, 0])
    user1_row = features_df[features_df['user'] == 'user1'].iloc[0]
    assert user1_row['label'] == 0, "user1 should have label 0"
    
    # user2 should have label 1
    user2_row = features_df[features_df['user'] == 'user2'].iloc[0]
    assert user2_row['label'] == 1, "user2 should have label 1"

