"""
Schema Detection and Dataset Inventory Script

Purpose: This script analyzes a CSV dataset to detect column types, generate a schema,
and provide insights about the data structure. This helps us understand what features
we can extract and what transformations are needed before training models.

How this file fits into the project:
- First step in the pipeline: understand your data before feature engineering
- Outputs a JSON schema that can be used by downstream scripts
- Provides human-readable summary for data exploration

Usage:
    python scripts/schema_and_inventory.py data/cert_dataset.csv
"""

import json
import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import re

def detect_column_type(series, col_name):
    """
    Infer the type of a column based on its values.
    
    This function uses heuristics to classify columns:
    - Timestamp: if values look like dates/times
    - IP: if values match IP address patterns
    - User: if column name suggests user identifiers
    - Label: if column name suggests binary classification target
    - Numeric: if all values are numbers
    - Categorical: otherwise (text fields with limited unique values)
    
    Args:
        series: pandas Series to analyze
        col_name: name of the column (helps with heuristics)
    
    Returns:
        str: detected type ('timestamp', 'numeric', 'categorical', 'ip', 'user', 'label')
    """
    # Convert to string for pattern matching
    sample_values = series.dropna().astype(str).head(100)
    
    if len(sample_values) == 0:
        return 'categorical'
    
    # Check for timestamp patterns
    # Common formats: YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, Unix timestamps
    timestamp_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{4}/\d{2}/\d{2}',   # YYYY/MM/DD
        r'\d{10,}',             # Unix timestamp (10+ digits)
    ]
    
    for pattern in timestamp_patterns:
        if sample_values.str.match(pattern).sum() / len(sample_values) > 0.5:
            # Try to parse as datetime to confirm
            try:
                pd.to_datetime(series.dropna().head(10))
                return 'timestamp'
            except:
                pass
    
    # Check for IP addresses
    ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
    if 'ip' in col_name.lower() or 'src' in col_name.lower() or 'dst' in col_name.lower():
        if sample_values.str.match(ip_pattern).sum() / len(sample_values) > 0.3:
            return 'ip'
    
    # Check for user identifiers
    if 'user' in col_name.lower() or 'id' in col_name.lower():
        return 'user'
    
    # Check for label columns
    if 'label' in col_name.lower() or 'target' in col_name.lower() or 'anomaly' in col_name.lower():
        return 'label'
    
    # Check if numeric
    try:
        pd.to_numeric(series.dropna().head(100))
        return 'numeric'
    except:
        pass
    
    # Default to categorical
    return 'categorical'

def generate_schema(df, output_path):
    """
    Generate a comprehensive schema JSON file from the dataframe.
    
    This schema includes:
    - Column names and detected types
    - Sample values for each column
    - Missing value counts
    - Unique value counts for categorical columns
    
    Args:
        df: pandas DataFrame to analyze
        output_path: path to save the JSON schema file
    """
    schema = {
        'columns': {},
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'generated_at': datetime.now().isoformat()
    }
    
    for col in df.columns:
        col_type = detect_column_type(df[col], col)
        
        # Get sample values (top 5 non-null)
        sample_values = df[col].dropna().head(5).tolist()
        # Convert to strings for JSON serialization
        sample_values = [str(v) for v in sample_values]
        
        schema['columns'][col] = {
            'type': col_type,
            'dtype': str(df[col].dtype),
            'missing_count': int(df[col].isna().sum()),
            'missing_percentage': float(df[col].isna().sum() / len(df) * 100),
            'unique_count': int(df[col].nunique()),
            'sample_values': sample_values
        }
        
        # Add extra info for numeric columns
        if col_type == 'numeric':
            schema['columns'][col]['min'] = float(df[col].min()) if not df[col].isna().all() else None
            schema['columns'][col]['max'] = float(df[col].max()) if not df[col].isna().all() else None
            schema['columns'][col]['mean'] = float(df[col].mean()) if not df[col].isna().all() else None
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    return schema

def print_schema_summary(schema):
    """
    Print a human-readable summary of the schema.
    
    This helps users quickly understand their dataset structure.
    """
    print("\n" + "="*80)
    print("DATASET SCHEMA SUMMARY")
    print("="*80)
    print(f"Total Rows: {schema['total_rows']:,}")
    print(f"Total Columns: {schema['total_columns']}")
    print(f"\nColumn Details:")
    print("-"*80)
    
    for col_name, col_info in schema['columns'].items():
        print(f"\nColumn: {col_name}")
        print(f"  Type: {col_info['type']}")
        print(f"  Missing: {col_info['missing_count']} ({col_info['missing_percentage']:.1f}%)")
        print(f"  Unique Values: {col_info['unique_count']}")
        
        if col_info['sample_values']:
            print(f"  Sample Values: {', '.join(col_info['sample_values'][:3])}")
        
        if col_info['type'] == 'numeric' and col_info.get('mean') is not None:
            print(f"  Range: [{col_info['min']:.2f}, {col_info['max']:.2f}], Mean: {col_info['mean']:.2f}")

def main():
    parser = argparse.ArgumentParser(
        description='Detect schema and generate inventory for CSV dataset'
    )
    parser.add_argument('input_csv', type=str, help='Path to input CSV file')
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/detected_schema.json',
        help='Path to output JSON schema file (default: data/detected_schema.json)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_csv).exists():
        print(f"Error: Input file not found: {args.input_csv}")
        sys.exit(1)
    
    print(f"Loading dataset from: {args.input_csv}")
    
    # Load CSV (try to read first 1000 rows for speed, then full dataset)
    try:
        df = pd.read_csv(args.input_csv, nrows=1000)
        print(f"Loaded {len(df)} rows for initial analysis...")
        
        # For schema detection, we can use a sample, but for full stats, load more
        if Path(args.input_csv).stat().st_size < 100 * 1024 * 1024:  # If < 100MB, load all
            df = pd.read_csv(args.input_csv)
            print(f"Loaded full dataset: {len(df):,} rows")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Generate schema
    print("\nAnalyzing columns and detecting types...")
    schema = generate_schema(df, args.output)
    
    # Print summary
    print_schema_summary(schema)
    
    print(f"\n[OK] Schema saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Review the schema to understand your data structure")
    print("  2. Run data_prep.py to generate features from this dataset")
    print("  3. Check for timestamp columns that need parsing")

if __name__ == '__main__':
    main()

