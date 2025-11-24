import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from preprocess import TextTokenizer
from model import DisasterBiLSTM
import os

def plot_individual_graphs():
    print("Generating Individual Model Visualizations...")
    
    # 0. Setup Output Directory
    output_dir = '../images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 1. Load Data
    try:
        df_text = pd.read_csv('../data/test.csv')
        df_labels = pd.read_csv('../data/test_label.csv')
        df = pd.merge(df_text, df_labels, on='id')
    except FileNotFoundError:
        print("Error: Files not found in data/ folder.")
        return

    # 2. Load Model & Vocab
    # weights_only=False suppresses the warning, assuming you created the file yourself
    checkpoint = torch.load('../saved_models/bilstm_model.pth', weights_only=False)
    vocab = checkpoint['vocab']
    
    tokenizer = TextTokenizer()
    tokenizer.word_index = vocab
    tokenizer.vocab_size = len(vocab) + 2
    
    X_seq = tokenizer.texts_to_sequences(df['text'].values)
    y_true = df['target'].values
    
    # 3. Predict Probabilities
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisasterBiLSTM(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Predicting on {device}...")
    with torch.no_grad():
        inputs = X_seq.to(device)
        outputs = model(inputs)
        y_probs = outputs.cpu().numpy().flatten()
        y_preds = (y_probs > 0.5).astype(int)

    # --- GRAPH 1: CONFUSION MATRIX ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Threshold=0.5)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Non-Disaster', 'Disaster'])
    plt.yticks([0.5, 1.5], ['Non-Disaster', 'Disaster'])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()
    print(f"Saved {output_dir}/confusion_matrix.png")

    # --- GRAPH 2: ROC CURVE ---
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate (False Alarms)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()
    print(f"Saved {output_dir}/roc_curve.png")

    # --- GRAPH 3: PRECISION-RECALL CURVE ---
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(recall, precision, color='green', lw=2)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_curve.png')
    plt.close()
    print(f"Saved {output_dir}/precision_recall_curve.png")

    # --- GRAPH 4: PROBABILITY HISTOGRAM ---
    plt.figure(figsize=(8, 6))
    sns.histplot(y_probs, bins=20, kde=True, color='purple')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Predicted Probability (0=Safe, 1=Disaster)')
    plt.ylabel('Count of Tweets')
    plt.axvline(0.5, color='red', linestyle='--', label='Threshold 0.5')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/probability_histogram.png')
    plt.close()
    print(f"Saved {output_dir}/probability_histogram.png")

if __name__ == "__main__":
    plot_individual_graphs()