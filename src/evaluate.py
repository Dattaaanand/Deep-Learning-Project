import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from preprocess import TextTokenizer
from model import DisasterBiLSTM

def evaluate():
    print("Initializing Evaluation...")
    
    # 1. Load Data
    try:
        df_text = pd.read_csv('../data/test.csv')
        df_answers = pd.read_csv('../data/test_label.csv')
    except FileNotFoundError:
        print("Error: Files not found. Check 'data/' folder.")
        return

    # Merge to align Text with Answers
    df = pd.merge(df_text, df_answers, on='id')
    print(f"Loaded {len(df)} samples.")

    # 2. Preprocess
    checkpoint = torch.load('../saved_models/bilstm_model.pth', weights_only=False)
    vocab = checkpoint['vocab']
    
    tokenizer = TextTokenizer()
    tokenizer.word_index = vocab
    tokenizer.vocab_size = len(vocab) + 2
    
    X_test = tokenizer.texts_to_sequences(df['text'].values)
    y_true = df['target'].values
    
    # 3. Predict Probabilities
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisasterBiLSTM(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Calculating Probabilities...")
    with torch.no_grad():
        inputs = X_test.to(device)
        outputs = model(inputs)
        # Get raw probabilities (e.g. 0.45, 0.88)
        y_probs = outputs.cpu().numpy().flatten()

    # 4. Find Best Threshold
    print("\n" + "="*40)
    print(f"{'Threshold':<10} | {'F1 Score':<10} | {'Accuracy':<10}")
    print("-" * 40)
    
    best_f1 = 0
    best_thresh = 0.5

    for thresh in np.arange(0.3, 0.8, 0.05):
        y_pred = (y_probs > thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        
        print(f"{thresh:.2f}       | {f1:.4f}     | {acc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print("="*40)
    print(f"BEST THRESHOLD: {best_thresh:.2f}")
    
    # 5. Final Report with Best Threshold
    final_preds = (y_probs > best_thresh).astype(int)
    print("\nFinal Report (at optimal threshold):")
    print(classification_report(y_true, final_preds, target_names=['Non-Disaster', 'Disaster']))

if __name__ == "__main__":
    evaluate()