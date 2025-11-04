"""
LSTM Inference Helper Script

Purpose: Use a trained LSTM model to make predictions on new sequences.

Usage:
    python scripts/lstm_infer.py --model models/lstm_model.pt --input data/features_test.csv
"""

import pandas as pd
import numpy as np
import torch
import argparse
from pathlib import Path
import joblib
from scripts.train_lstm import LSTMModel, UserSequenceDataset
from torch.utils.data import DataLoader

def load_model(model_path, device):
    """Load trained LSTM model."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    model = LSTMModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['feature_names']

def main():
    parser = argparse.ArgumentParser(description='LSTM inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--scaler', type=str, default='models/lstm_scaler.pkl', help='Path to scaler')
    parser.add_argument('--input', type=str, required=True, help='Input features CSV')
    parser.add_argument('--sequence_length', type=int, default=7, help='Sequence length')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, feature_names = load_model(args.model, device)
    
    # Load scaler
    scaler = joblib.load(args.scaler)
    
    # Load data and create sequences
    df = pd.read_csv(args.input)
    # Implementation would create sequences similar to train_lstm.py
    # This is a simplified version
    
    print("LSTM inference script - use trained model for predictions")

if __name__ == '__main__':
    main()

