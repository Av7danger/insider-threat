"""
Unit Tests for Schema Detection

Purpose: Test that schema_and_inventory.py correctly detects column types and generates schemas.

How to run:
    pytest tests/test_schema.py -v
"""

import pytest
import pandas as pd
import json
from pathlib import Path
import tempfile
import sys

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from schema_and_inventory import detect_column_type, generate_schema

def test_detect_column_type_timestamp():
    """Test timestamp detection."""
    dates = pd.Series(['2020-01-01', '2020-01-02', '2020-01-03'])
    assert detect_column_type(dates, 'date') == 'timestamp'

def test_detect_column_type_numeric():
    """Test numeric detection."""
    numbers = pd.Series([1, 2, 3, 4, 5])
    assert detect_column_type(numbers, 'count') == 'numeric'

def test_detect_column_type_categorical():
    """Test categorical detection."""
    categories = pd.Series(['A', 'B', 'C', 'A', 'B'])
    assert detect_column_type(categories, 'category') == 'categorical'

def test_generate_schema():
    """Test schema generation from sample data."""
    # Create sample dataframe
    df = pd.DataFrame({
        'user': ['user1', 'user2', 'user3'],
        'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'count': [10, 20, 30],
        'category': ['A', 'B', 'A']
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        schema_path = f.name
    
    try:
        schema = generate_schema(df, schema_path)
        
        # Check schema structure
        assert 'columns' in schema
        assert 'total_rows' in schema
        assert schema['total_rows'] == 3
        
        # Check column detection
        assert 'user' in schema['columns']
        assert 'date' in schema['columns']
        assert 'count' in schema['columns']
        
        # Verify JSON file was created
        assert Path(schema_path).exists()
        
        # Verify JSON is valid
        with open(schema_path) as f:
            loaded = json.load(f)
        assert loaded['total_rows'] == 3
        
    finally:
        Path(schema_path).unlink()

def test_schema_with_missing_values():
    """Test schema generation with missing values."""
    df = pd.DataFrame({
        'col1': [1, 2, None, 4, 5],
        'col2': ['A', 'B', 'C', None, 'E']
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        schema_path = f.name
    
    try:
        schema = generate_schema(df, schema_path)
        
        # Check missing value counts
        assert schema['columns']['col1']['missing_count'] == 1
        assert schema['columns']['col2']['missing_count'] == 1
        
    finally:
        Path(schema_path).unlink()

