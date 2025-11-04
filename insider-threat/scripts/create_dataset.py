"""
Quick Dataset Creator - Run this after installing dependencies

Usage: python scripts/create_dataset.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def main():
    random.seed(42)
    np.random.seed(42)
    
    users = [f'user{i:03d}' for i in range(1, 21)]
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(30)]
    base_ips = [f'192.168.1.{i}' for i in range(10, 50)]
    external_ips = [f'10.0.0.{i}' for i in range(1, 20)]
    file_paths_normal = [f'/home/{u}/file{j}.txt' for u in users[:15] for j in range(1, 11)]
    file_paths_normal += [f'/shared/docs/doc{j}.pdf' for j in range(1, 21)]
    file_paths_sensitive = [
        '/confidential/salary_data.csv',
        '/confidential/employee_records.db',
        '/confidential/contracts/contract_2020.pdf',
        '/confidential/financial/report.xlsx',
        '/confidential/customer_data.csv'
    ]
    
    data = []
    
    # Normal events (475 rows)
    for _ in range(475):
        user = np.random.choice(users)
        date = np.random.choice(dates)
        hour = np.random.choice([9, 10, 11, 12, 13, 14, 15, 16, 17])
        timestamp = date.replace(hour=hour, minute=np.random.randint(0, 60))
        data.append({
            'user': user,
            'date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': np.random.choice(base_ips),
            'dst_ip': np.random.choice(base_ips[:20]),
            'file_path': np.random.choice(file_paths_normal),
            'success': np.random.choice([1, 1, 1, 1, 0]),
            'label': 0
        })
    
    # Anomalous events (25 rows)
    for _ in range(25):
        user = np.random.choice(users[-5:])
        date = np.random.choice(dates)
        hour = np.random.choice([0, 1, 2, 3, 4, 22, 23] if np.random.random() < 0.5 else [9, 10, 11, 12, 13, 14, 15, 16, 17])
        timestamp = date.replace(hour=hour, minute=np.random.randint(0, 60))
        src_ip = np.random.choice(external_ips if np.random.random() < 0.7 else base_ips)
        dst_ip = np.random.choice(base_ips + external_ips)
        file_path = np.random.choice(file_paths_sensitive if np.random.random() < 0.6 else file_paths_normal)
        data.append({
            'user': user,
            'date': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'file_path': file_path,
            'success': np.random.choice([1, 0, 0, 0, 0]),
            'label': 1
        })
    
    random.shuffle(data)
    df = pd.DataFrame(data)
    df.to_csv('data/cert_dataset.csv', index=False)
    print(f'[OK] Generated {len(df)} rows')
    print(f'  Anomalies: {df["label"].sum()} ({df["label"].mean()*100:.1f}%)')

if __name__ == '__main__':
    main()

