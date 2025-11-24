import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from preprocess import TextTokenizer
from model import DisasterBiLSTM
import os

def evaluate():
    print("Initializing Evaluation...")
    
    # 1. Load Data
    # We need 'test.csv' for the Tweet Text and 'test_labeled.csv' for the Answers
    try:
        df_text = pd.read_csv('../data/test.csv')
        df_answers = pd.read_csv('../data/test_labeled.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        print("Ensure 'test.csv' and 'test_labeled.csv' are inside the 'data/' folder.")
        return

    # Merge them on 'id' to make sure the text matches the correct answer
    df = pd.merge(df_text, df_answers, on='id')
    print(f"Loaded {len(df)} samples for evaluation.")

    # 2. Prepare Tokenizer (using vocab from saved model)
    checkpoint = torch.load('../saved_models/bilstm_model.pth', weights_only=False)
    vocab = checkpoint['vocab']
    
    tokenizer = TextTokenizer()
    tokenizer.word_index = vocab
    tokenizer.vocab_size = len(vocab) + 2
    
    # Convert text to sequences
    X_test = tokenizer.texts_to_sequences(df['text'].values)
    y_true = df['target'].values
    
    # 3. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisasterBiLSTM(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. Run Predictions
    print(f"Predicting using {device}...")
    with torch.no_grad():
        inputs = X_test.to(device)
        outputs = model(inputs)
        # Convert probabilities (e.g. 0.89) to class labels (1)
        y_pred = (outputs > 0.5).int().cpu().numpy().flatten()

    # 5. Print Report
    print("\n" + "="*40)
    print("FINAL MODEL PERFORMANCE")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print("-" * 40)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-Disaster', 'Disaster']))

if __name__ == "__main__":
    evaluate()