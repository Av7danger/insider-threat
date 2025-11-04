"""
Release Notes Generator

Purpose: Automatically generate release notes from evaluation metrics and model information.
This helps create consistent, informative release notes for GitHub releases.

Usage:
    python scripts/create_release_notes.py --output docs/release_notes.md
"""

import pandas as pd
import argparse
import hashlib
from pathlib import Path
from datetime import datetime

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def generate_release_notes(metrics_path, output_path):
    """
    Generate release notes from metrics and model information.
    
    Args:
        metrics_path: Path to summary_metrics.csv
        output_path: Path to save release notes
    """
    notes = []
    notes.append("# Release Notes")
    notes.append("")
    notes.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    notes.append("")
    notes.append("## Model Performance")
    notes.append("")
    
    # Load metrics if available
    if Path(metrics_path).exists():
        metrics_df = pd.read_csv(metrics_path)
        notes.append("### Evaluation Metrics")
        notes.append("")
        notes.append(metrics_df.to_markdown(index=False))
        notes.append("")
    else:
        notes.append("⚠ Metrics file not found. Train and evaluate models first.")
        notes.append("")
    
    # Model file information
    models_dir = Path('models')
    if models_dir.exists():
        notes.append("## Model Files")
        notes.append("")
        
        model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.pt'))
        if model_files:
            notes.append("| Model File | Size | SHA256 Hash |")
            notes.append("|------------|------|-------------|")
            
            for model_file in sorted(model_files):
                size = model_file.stat().st_size / (1024 * 1024)  # MB
                file_hash = calculate_file_hash(model_file)
                notes.append(f"| {model_file.name} | {size:.2f} MB | {file_hash[:16]}... |")
        else:
            notes.append("No model files found.")
        
        notes.append("")
    
    # Save release notes
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(notes))
    
    print(f"✓ Release notes saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate release notes')
    parser.add_argument(
        '--metrics',
        type=str,
        default='artifacts/summary_metrics.csv',
        help='Path to summary metrics CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='docs/release_notes.md',
        help='Path to save release notes'
    )
    
    args = parser.parse_args()
    
    generate_release_notes(args.metrics, args.output)

if __name__ == '__main__':
    main()

