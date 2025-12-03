"""
Large Synthetic Insider Threat Dataset Generator

Creates a big CERT-style event log CSV suitable for demos and training.
The data is:
- Random but structured: users, dates, internal/external IPs, normal & sensitive files
- Contains multiple anomaly patterns
- Same schema as `data/cert_dataset.csv` / `sample_cert_small.csv`:
    user,date,src_ip,dst_ip,file_path,success,label

Usage (from project root *or* insider-threat directory):
    python scripts/create_large_dataset.py
    python scripts/create_large_dataset.py --rows 500000 --anomaly-rate 0.03
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    rows: int = 300_000          # total number of events
    anomaly_rate: float = 0.03   # fraction of events labelled as anomalies
    seed: int = 42
    users: int = 200             # number of distinct users
    days: int = 60               # number of days in the simulated period


SENSITIVE_PATHS = [
    "/confidential/salary_data.csv",
    "/confidential/employee_records.db",
    "/confidential/contracts/contract_2025.pdf",
    "/confidential/financial/report_q1.xlsx",
    "/confidential/customer_data.csv",
]


def make_users(n: int) -> List[str]:
    return [f"user{i:03d}" for i in range(1, n + 1)]


def make_dates(start: datetime, days: int) -> List[datetime]:
    return [start + timedelta(days=i) for i in range(days)]


def make_ips() -> Tuple[List[str], List[str]]:
    internal = [f"192.168.1.{i}" for i in range(10, 250)]
    external = [f"10.0.0.{i}" for i in range(1, 250)]
    return internal, external


def make_file_paths(users: List[str]) -> Tuple[List[str], List[str]]:
    normal: List[str] = []
    for u in users:
        for j in range(1, 21):
            normal.append(f"/home/{u}/file{j}.txt")
    for j in range(1, 101):
        normal.append(f"/shared/docs/doc{j}.pdf")
    return normal, SENSITIVE_PATHS


def sample_timestamp(base_date: datetime, work_hours: bool, rng: np.random.Generator) -> datetime:
    if work_hours:
        hour = rng.integers(8, 18)  # 08–17
    else:
        # off-hours / weekend style
        hour = int(rng.choice([*range(0, 7), *range(20, 24)]))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return base_date.replace(hour=hour, minute=minute, second=second)


def generate_dataset(cfg: Config) -> pd.DataFrame:
    random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    users = make_users(cfg.users)
    dates = make_dates(datetime(2020, 1, 1), cfg.days)
    internal_ips, external_ips = make_ips()
    normal_files, sensitive_files = make_file_paths(users)

    total_rows = cfg.rows
    anomaly_rows = int(total_rows * cfg.anomaly_rate)
    normal_rows = total_rows - anomaly_rows

    records: List[dict] = []

    # ----- Normal behaviour -----
    for _ in range(normal_rows):
        user = rng.choice(users)
        date = rng.choice(dates)
        ts = sample_timestamp(date, work_hours=True, rng=rng)

        records.append(
            {
                "user": user,
                "date": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip": rng.choice(internal_ips),
                "dst_ip": rng.choice(internal_ips),
                "file_path": rng.choice(normal_files),
                # mostly successful, with small error rate
                "success": int(rng.choice([1, 1, 1, 1, 1, 0])),
                "label": 0,
            }
        )

    # ----- Anomalous behaviour patterns -----
    # Mix several patterns for realism
    anomaly_patterns = ["off_hours", "external_src", "sensitive_files", "burst_access"]

    for _ in range(anomaly_rows):
        user = rng.choice(users[-max(10, cfg.users // 5) :])  # concentrate anomalies in a subset
        date = rng.choice(dates)

        pattern = rng.choice(anomaly_patterns)

        if pattern == "off_hours":
            ts = sample_timestamp(date, work_hours=False, rng=rng)
            src_ip = rng.choice(internal_ips)
            dst_ip = rng.choice(internal_ips)
            file_path = rng.choice(normal_files + sensitive_files)
        elif pattern == "external_src":
            ts = sample_timestamp(date, work_hours=False, rng=rng)
            src_ip = rng.choice(external_ips)
            dst_ip = rng.choice(internal_ips + external_ips)
            file_path = rng.choice(normal_files + sensitive_files)
        elif pattern == "sensitive_files":
            ts = sample_timestamp(date, work_hours=rng.random() < 0.4, rng=rng)
            src_ip = rng.choice(internal_ips + external_ips)
            dst_ip = rng.choice(internal_ips + external_ips)
            file_path = rng.choice(sensitive_files)
        else:  # burst_access
            ts = sample_timestamp(date, work_hours=False, rng=rng)
            src_ip = rng.choice(internal_ips + external_ips)
            dst_ip = rng.choice(internal_ips)
            # simulate lots of different shared docs
            file_path = rng.choice(
                normal_files
                + sensitive_files
                + [f"/shared/bulk_export/report_{i}.csv" for i in range(1, 51)]
            )

        success = int(rng.choice([0, 0, 0, 1]))  # more failures

        records.append(
            {
                "user": user,
                "date": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "file_path": file_path,
                "success": success,
                "label": 1,
            }
        )

    # Shuffle records for realism
    df = pd.DataFrame.from_records(records)
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a large synthetic CERT-style dataset.")
    parser.add_argument("--rows", type=int, default=300_000, help="Total number of rows to generate.")
    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.03,
        help="Fraction of rows that are labelled anomalies (0–1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    cfg = Config(rows=args.rows, anomaly_rate=args.anomaly_rate, seed=args.seed)

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "cert_dataset_large.csv"

    df = generate_dataset(cfg)
    df.to_csv(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Wrote large dataset to: {output_path}")
    print(f"     Rows: {len(df)}")
    print(f"     Anomalies: {int(df['label'].sum())} "
          f"({df['label'].mean() * 100:.2f}%)")
    print(f"     File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()


