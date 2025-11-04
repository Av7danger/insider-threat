"""
LSTM Training Script

Purpose: Train a sequence model using PyTorch LSTM to detect anomalies in user behavior
over time. Unlike tabular models (XGBoost, Isolation Forest), LSTM can learn temporal
patterns - it "remembers" what a user did in previous days and predicts if today's
behavior is anomalous.

What LSTM does (in plain words):
- LSTM = Long Short-Term Memory, a type of neural network
- Reads sequences of data (e.g., last 7 days of user activity)
- Learns patterns like "user usually logs in at 9 AM, but today at 3 AM"
- Good at detecting gradual changes in behavior or unusual sequences

Sequence construction:
- For each user, we create sliding windows of N days
- Each window becomes one training example
- Example: if sequence_length=7, we use days 1-7 to predict day 8's label
- This captures temporal dependencies

Shapes explained:
- batch_size: how many sequences to process at once (larger = faster but more memory)
- seq_len: how many days to look back (sequence_length)
- features: number of features per day (e.g., total_events, unique_src_ip, etc.)

GPU vs CPU:
- GPU is much faster for training neural networks (10-100x speedup)
- CPU is fine for small datasets or testing
- This script auto-detects GPU (CUDA) if available

How this file fits into the project:
- Alternative approach to XGBoost that captures temporal patterns
- Useful when user behavior changes gradually over time
- Requires labeled data (supervised learning)

Usage:
    python scripts/train_lstm.py --input data/cert_dataset.csv --epochs 10
"""

import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class UserSequenceDataset(Dataset):
    """
    PyTorch Dataset for user behavior sequences.
    
    This class creates sequences of user activity over time.
    Each sequence is a sliding window of N days used to predict the next day's label.
    """
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class LSTMModel(nn.Module):
    """
    Simple LSTM model for sequence classification.
    
    Architecture:
    - LSTM layer: processes sequences
    - Dropout: prevents overfitting
    - Fully connected layer: final classification
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)  # Binary classification
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output

def create_sequences(df, user_col, date_col, feature_cols, label_col, sequence_length=7):
    """
    Create sequences from user-day aggregated features.
    
    For each user, we create sliding windows of N consecutive days.
    Each sequence is used to predict the label of the day after the sequence.
    
    Args:
        df: dataframe with user, date, features, label
        user_col: column name for user identifier
        date_col: column name for date
        feature_cols: list of feature column names
        label_col: column name for label
        sequence_length: number of days in each sequence
    
    Returns:
        sequences: array of shape (n_samples, sequence_length, n_features)
        labels: array of shape (n_samples,)
    """
    sequences = []
    labels = []
    
    # Sort by user and date
    df = df.sort_values([user_col, date_col]).reset_index(drop=True)
    
    # Get unique users
    users = df[user_col].unique()
    
    logger.info(f"Creating sequences for {len(users)} users...")
    
    for user in users:
        user_df = df[df[user_col] == user].sort_values(date_col).reset_index(drop=True)
        
        # Need at least sequence_length + 1 days (sequence + prediction target)
        if len(user_df) < sequence_length + 1:
            continue
        
        # Create sliding windows
        for i in range(len(user_df) - sequence_length):
            # Get sequence of days
            sequence = user_df.iloc[i:i+sequence_length]
            # Get label for the day after the sequence
            target_label = user_df.iloc[i+sequence_length][label_col]
            
            # Extract features
            seq_features = sequence[feature_cols].values
            sequences.append(seq_features)
            labels.append(int(target_label))
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    logger.info(f"Created {len(sequences)} sequences")
    logger.info(f"Sequence shape: {sequences.shape}")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    
    return sequences, labels

def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    """
    Train the LSTM model.
    
    Returns training history (losses) for plotting.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description='Train LSTM sequence model')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (features or raw data)'
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default='models/lstm_model.pt',
        help='Path to save trained model (default: models/lstm_model.pt)'
    )
    parser.add_argument(
        '--output_scaler',
        type=str,
        default='models/lstm_scaler.pkl',
        help='Path to save scaler (default: models/lstm_scaler.pkl)'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=7,
        help='Number of days in each sequence (default: 7)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=64,
        help='LSTM hidden size (default: 64)'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=2,
        help='Number of LSTM layers (default: 2)'
    )
    parser.add_argument(
        '--label_col',
        type=str,
        default='label',
        help='Name of label column (default: label)'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} rows")
    
    # Check if we have features or need to prepare them
    if 'total_events' not in df.columns:
        logger.warning("Features not found. This script expects aggregated features.")
        logger.warning("Please run data_prep.py first to generate features.")
        return
    
    # Find columns
    user_col = 'user' if 'user' in df.columns else df.columns[0]
    date_col = 'date' if 'date' in df.columns else df.columns[1]
    
    if args.label_col not in df.columns:
        logger.error(f"Label column '{args.label_col}' not found. LSTM requires labeled data.")
        return
    
    # Get feature columns
    exclude_cols = ['user', 'date', 'label', 'target', 'anomaly']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Using {len(feature_cols)} features")
    
    # Create sequences
    sequences, labels = create_sequences(
        df, user_col, date_col, feature_cols, args.label_col, args.sequence_length
    )
    
    # Scale features
    logger.info("Scaling features...")
    n_samples, seq_len, n_features = sequences.shape
    sequences_reshaped = sequences.reshape(-1, n_features)
    scaler = StandardScaler()
    sequences_scaled = scaler.fit_transform(sequences_reshaped)
    sequences_scaled = sequences_scaled.reshape(n_samples, seq_len, n_features)
    
    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        sequences_scaled, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Train sequences: {len(X_train)}, Val sequences: {len(X_val)}")
    
    # Create data loaders
    train_dataset = UserSequenceDataset(X_train, y_train)
    val_dataset = UserSequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = LSTMModel(
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    logger.info("Training LSTM model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, args.epochs, args.learning_rate, device
    )
    
    # Save model
    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': n_features,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'sequence_length': args.sequence_length
        },
        'feature_names': feature_cols
    }, output_model_path)
    logger.info(f"✓ Saved model to: {output_model_path}")
    
    # Save scaler
    import joblib
    output_scaler_path = Path(args.output_scaler)
    output_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_scaler_path)
    logger.info(f"✓ Saved scaler to: {output_scaler_path}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Training Curve')
    plt.legend()
    plt.grid(True)
    
    curve_path = Path('artifacts/lstm_training_curve.png')
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(curve_path)
    logger.info(f"✓ Saved training curve to: {curve_path}")
    plt.close()
    
    logger.info("\nNext steps:")
    logger.info("  1. Use lstm_infer.py for inference on new sequences")
    logger.info("  2. Compare with XGBoost and Isolation Forest models")

if __name__ == '__main__':
    main()

