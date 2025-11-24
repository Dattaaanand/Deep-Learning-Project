import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from preprocess import TextTokenizer
from model import DisasterBiLSTM
import os

# Configuration
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
PATIENCE = 5

def train_model():
    # 1. Load Data (Augmented)
    print("Loading augmented data...")
    try:
        df_train = pd.read_csv('../data/train_augmented.csv')
        print(f"Training on {len(df_train)} samples.")
    except FileNotFoundError:
        print("Error: 'train_augmented.csv' not found. Did you run augment.py?")
        return
    
    # 2. Prepare Tokenizer and Sequences
    tokenizer = TextTokenizer()
    tokenizer.fit(df_train['text'].values)
    
    X = tokenizer.texts_to_sequences(df_train['text'].values)
    y = torch.tensor(df_train['target'].values, dtype=torch.float32).unsqueeze(1)
    
    # 3. Stratified Split (80/20)
    # Since augment.py shuffled the data, a simple index split is random enough
    split_idx = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split_idx], y[:split_idx])
    val_ds = TensorDataset(X[split_idx:], y[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # 4. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisasterBiLSTM(vocab_size=tokenizer.vocab_size).to(device)
    
    # 5. Optimizer and Loss
    # FIX: Reduced weight_decay from 1e-3 back to 1e-5 to allow learning
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Scheduler to fine-tune learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    criterion = nn.BCELoss(reduction='none') 

    # Class Weights: {0: 0.87, 1: 1.15}
    def weighted_loss(predictions, targets):
        loss = criterion(predictions, targets)
        weights = targets * 1.15 + (1 - targets) * 0.87
        return (loss * weights).mean()

    # 6. Training Loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = weighted_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = weighted_loss(outputs, targets)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': tokenizer.word_index
            }, '../saved_models/bilstm_model.pth')
            print("Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    train_model()