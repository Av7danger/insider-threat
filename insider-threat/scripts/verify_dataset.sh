#!/bin/bash
# Dataset Verification Script
# 
# Purpose: Quickly verify that the dataset file exists and show basic stats
# This helps catch missing files early before running expensive scripts

DATASET_PATH="data/cert_dataset.csv"

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset file not found at $DATASET_PATH"
    echo ""
    echo "Please ensure the dataset CSV file is placed at: $DATASET_PATH"
    echo "If you don't have the full dataset, you can use the sample file:"
    echo "  data/sample_cert_small.csv"
    exit 1
fi

echo "✓ Dataset file found: $DATASET_PATH"
echo ""

# Count rows and columns (using Python for cross-platform compatibility)
python3 << EOF
import pandas as pd
import sys

try:
    df = pd.read_csv("$DATASET_PATH", nrows=1000000)  # Read up to 1M rows for speed
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns[:10], 1):
        print(f"  {i}. {col}")
    if len(df.columns) > 10:
        print(f"  ... and {len(df.columns) - 10} more columns")
    
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Dataset verification complete"
    exit 0
else
    echo ""
    echo "❌ Dataset verification failed"
    exit 1
fi

