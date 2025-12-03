"""
Create a smaller, higher-quality sample dataset for the demo app.

Reads the main synthetic CERT-style dataset created by `create_dataset.py`
and writes a new CSV with:
- Same columns as `data/cert_dataset.csv`
- More balanced labels (higher proportion of anomalies)
- Shuffled rows

Usage (from project root or insider-threat directory):
    python scripts/create_better_sample.py
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    source_path = data_dir / "cert_dataset.csv"
    output_path = data_dir / "sample_cert_better.csv"

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source dataset not found at {source_path}. "
            "Run `python scripts/create_dataset.py` first."
        )

    df = pd.read_csv(source_path)

    if "label" not in df.columns:
        raise ValueError("Source dataset must contain a 'label' column.")

    # Separate normal and anomalous events
    normal = df[df["label"] == 0]
    anomalous = df[df["label"] == 1]

    # Pick a reasonably sized sample:
    # - up to 200 normal rows
    # - up to all anomalies, but cap at 100 to avoid huge files
    n_norm = min(len(normal), 200)
    n_anom = min(len(anomalous), 100)

    normal_sample = normal.sample(n=n_norm, random_state=42)
    anomalous_sample = anomalous.sample(n=n_anom, random_state=42)

    sample = pd.concat([normal_sample, anomalous_sample], axis=0)

    # Shuffle rows so anomalies are mixed in
    sample = sample.sample(frac=1.0, random_state=42).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)

    print(f"[OK] Wrote better sample dataset to: {output_path}")
    print(f"     Rows: {len(sample)}")
    print(f"     Anomalies: {int(sample['label'].sum())} "
          f"({sample['label'].mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()


