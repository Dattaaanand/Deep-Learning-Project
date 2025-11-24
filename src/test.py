import torch
import pandas as pd
from preprocess import TextTokenizer
from model import DisasterBiLSTM

def predict():
    # 1. Load Data
    df_test = pd.read_csv('../data/test.csv')
    print(f"Loaded {len(df_test)} test tweets.")
    
    # 2. Load Model and Vocab
    checkpoint = torch.load('../saved_models/bilstm_model.pth')
    vocab = checkpoint['vocab']
    
    # Re-initialize Tokenizer with saved vocab
    tokenizer = TextTokenizer()
    tokenizer.word_index = vocab
    tokenizer.vocab_size = len(vocab) + 2
    
    # 3. Preprocess Test Data
    X_test = tokenizer.texts_to_sequences(df_test['text'].values)
    
    # 4. Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DisasterBiLSTM(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 5. Prediction
    print("Predicting...")
    predictions = []
    with torch.no_grad():
        # Process in batches if dataset is large, here doing all at once for simplicity
        inputs = X_test.to(device)
        outputs = model(inputs)
        preds = (outputs > 0.5).int().cpu().numpy().flatten()
        predictions = preds
        
    # 6. Save Results
    submission = pd.DataFrame({
        'id': df_test['id'],
        'target': predictions
    })
    submission.to_csv('../outputs/prediction.csv', index=False)
    print("Predictions saved to ../prediction.csv")

if __name__ == "__main__":
    predict()