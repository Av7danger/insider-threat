"""
Master Script - Run Everything!
This script runs the complete Insider Threat Detection project from start to finish:
1. Checks environment
2. Generates dataset
3. Runs full ML pipeline
4. Tests everything
5. Optionally starts the API server

Usage:
    python run_all.py              # Run everything except API
    python run_all.py --api        # Run everything including starting API
    python run_all.py --skip-tests # Skip tests for faster run
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import argparse

def print_header(text, color_code=""):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_step(step_num, total, text):
    """Print step header"""
    print(f"\n[{step_num}/{total}] {text}")
    print("-" * 60)

def run_command(cmd_list, description=""):
    """Run a command and show output"""
    try:
        result = subprocess.run(
            cmd_list,
            shell=False,
            capture_output=True,
            text=True,
            check=False,
            cwd=os.getcwd()
        )
        
        # Print relevant output
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in [
                    'generated', 'saved', 'training', 'accuracy', 
                    'roc-auc', 'precision', 'recall', 'passed', 
                    'features shape', 'anomalies detected', '[ok]',
                    'complete', 'success'
                ]):
                    print(f"  {line}")
        
        if result.returncode != 0:
            print(f"  [WARNING] {description} had issues (check output above)")
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[-3:]:  # Last 3 lines of error
                    if line.strip():
                        print(f"  ERROR: {line}")
            return False
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def check_environment():
    """Check if environment is set up correctly"""
    print_step(0, 10, "Checking Environment...")
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check virtual environment
    venv_python = script_dir / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = script_dir / "venv" / "bin" / "python"
    
    if not venv_python.exists():
        print("  [ERROR] Virtual environment not found!")
        print("  Please run: python -m venv venv")
        print("  Then: pip install -r requirements.txt")
        return None, None
    
    python_cmd = str(venv_python)
    
    # Check key dependencies
    print("  Checking dependencies...")
    result = subprocess.run(
        [python_cmd, "-c", "import pandas, numpy, sklearn, xgboost, fastapi"],
        capture_output=True
    )
    
    if result.returncode != 0:
        print("  [WARNING] Some dependencies missing. Installing...")
        print("  Run: pip install -r requirements.txt")
        return None, None
    
    print("  [OK] Environment ready!")
    return python_cmd, script_dir

def main():
    parser = argparse.ArgumentParser(description='Run complete Insider Threat Detection pipeline')
    parser.add_argument('--api', action='store_true', help='Start API server after pipeline')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset generation (use existing)')
    
    args = parser.parse_args()
    
    print_header("INSIDER THREAT DETECTION - COMPLETE PIPELINE")
    
    # Check environment
    python_cmd, script_dir = check_environment()
    if not python_cmd:
        sys.exit(1)
    
    steps_completed = 0
    total_steps = 8 if not args.skip_tests else 7
    if args.api:
        total_steps += 1
    
    # Step 1: Generate Dataset
    if not args.skip_dataset:
        print_step(1, total_steps, "Step 1: Generating Synthetic Dataset...")
        if run_command([python_cmd, 'scripts/create_dataset.py'], "Dataset generation"):
            steps_completed += 1
        else:
            print("  Continuing anyway...")
    else:
        print_step(1, total_steps, "Step 1: Using Existing Dataset...")
        if Path("data/cert_dataset.csv").exists():
            print("  [OK] Dataset found")
            steps_completed += 1
        else:
            print("  [ERROR] Dataset not found!")
            sys.exit(1)
    
    # Step 2: Schema Detection
    print_step(2, total_steps, "Step 2: Detecting Schema...")
    run_command([python_cmd, 'scripts/schema_and_inventory.py', 'data/cert_dataset.csv'], "Schema detection")
    steps_completed += 1
    
    # Step 3: Feature Engineering
    print_step(3, total_steps, "Step 3: Engineering Features...")
    if run_command([
        python_cmd, 'scripts/data_prep.py',
        '--input', 'data/cert_dataset.csv',
        '--output', 'data/features.csv',
        '--split'
    ], "Feature engineering"):
        steps_completed += 1
    else:
        print("  [ERROR] Feature engineering failed!")
        sys.exit(1)
    
    # Step 4: Train Isolation Forest
    print_step(4, total_steps, "Step 4: Training Isolation Forest...")
    if run_command([
        python_cmd, 'scripts/train_iso.py',
        '--input', 'data/features_train.csv',
        '--contamination', '0.01'
    ], "Isolation Forest training"):
        steps_completed += 1
    else:
        print("  [ERROR] Isolation Forest training failed!")
        sys.exit(1)
    
    # Step 5: Train XGBoost
    print_step(5, total_steps, "Step 5: Training XGBoost...")
    if run_command([
        python_cmd, 'scripts/train_xgb.py',
        '--input', 'data/features_train.csv',
        '--test_path', 'data/features_test.csv'
    ], "XGBoost training"):
        steps_completed += 1
    else:
        print("  [WARNING] XGBoost training failed (may not have labels)")
    
    # Step 6: Evaluate Models
    print_step(6, total_steps, "Step 6: Evaluating Models...")
    if run_command([
        python_cmd, 'scripts/evaluate.py',
        '--test_data', 'data/features_test.csv'
    ], "Model evaluation"):
        steps_completed += 1
    
    # Step 7: Run Tests
    if not args.skip_tests:
        print_step(7, total_steps, "Step 7: Running Tests...")
        run_command([python_cmd, '-m', 'pytest', 'tests/', '-v', '--tb=short'], "Tests")
        steps_completed += 1
    
    # Step 8: Summary
    print_step(total_steps, total_steps, "Generating Summary...")
    
    print_header("PIPELINE COMPLETE!")
    
    # Count files
    models_dir = Path("models")
    artifacts_dir = Path("artifacts")
    data_dir = Path("data")
    
    model_count = len(list(models_dir.glob("*.pkl"))) if models_dir.exists() else 0
    artifact_count = len([f for f in artifacts_dir.glob("*") if f.suffix in ['.csv', '.json', '.png', '.md']]) if artifacts_dir.exists() else 0
    data_count = len(list(data_dir.glob("*.csv"))) if data_dir.exists() else 0
    
    print(f"\nGenerated Files:")
    print(f"  ✓ Models: {model_count} files")
    print(f"  ✓ Artifacts: {artifact_count} files")
    print(f"  ✓ Data Files: {data_count} files")
    print(f"\nSteps Completed: {steps_completed}/{total_steps}")
    
    # Show key metrics if available
    metrics_file = Path("artifacts/summary_metrics.csv")
    if metrics_file.exists():
        print(f"\n📊 Key Metrics (check artifacts/summary_metrics.csv for details):")
        try:
            import pandas as pd
            df = pd.read_csv(metrics_file)
            for _, row in df.iterrows():
                model_name = row['model']
                if 'roc_auc' in row and pd.notna(row['roc_auc']):
                    print(f"  {model_name}: ROC-AUC = {row['roc_auc']:.3f}")
        except:
            pass
    
    # Start API if requested
    if args.api:
        print_header("Starting API Server...")
        print("\nAPI will be available at: http://localhost:8000")
        print("Interactive docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the server")
        print("\nStarting server...")
        try:
            subprocess.run([
                python_cmd, '-m', 'uvicorn',
                'app.inference_api:app',
                '--host', '0.0.0.0',
                '--port', '8000'
            ])
        except KeyboardInterrupt:
            print("\n\nAPI server stopped.")
    else:
        print("\n" + "="*60)
        print("Next Steps:")
        print("="*60)
        print("  1. Review results: artifacts/summary_metrics.csv")
        print("  2. View visualizations: artifacts/*.png")
        print("  3. Start API: python run_all.py --api")
        print("  4. Or manually: python -m uvicorn app.inference_api:app --host 0.0.0.0 --port 8000")
        print("  5. Open in browser: http://localhost:8000/docs")
        print()

if __name__ == "__main__":
    main()

