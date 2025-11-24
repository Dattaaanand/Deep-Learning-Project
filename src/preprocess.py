import re
import torch
import numpy as np
from collections import Counter

# Text Cleaning based on PDF "Text Cleaning (One-pass Pipeline)" [cite: 86]
def clean_text(text):
    text = text.lower() # [cite: 89]
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # URLs [cite: 90]
    text = re.sub(r'@\w+', '', text) # Mentions (interpreted from regex intent) [cite: 91]
    text = re.sub(r'#(\w+)', r'\1', text) # Hashtags (keep word) [cite: 92]
    text = re.sub(r'[^a-zA-Z0-9\s\']', '', text) # Punctuation [cite: 95]
    text = re.sub(r'\s+', ' ', text).strip() # Whitespace [cite: 96]
    return text

class TextTokenizer:
    def __init__(self, max_words=10000, max_len=50):
        self.max_words = max_words # [cite: 100]
        self.max_len = max_len     # [cite: 101]
        self.word_index = {}
        self.vocab_size = 0

    def fit(self, texts):
        # Build vocab from top 10,000 words
        all_words = []
        for text in texts:
            all_words.extend(clean_text(text).split())
        
        count = Counter(all_words)
        # 0 is reserved for padding, 1 for OOV (Out of Vocabulary)
        common_words = count.most_common(self.max_words - 2)
        
        self.word_index = {word: i + 2 for i, (word, _) in enumerate(common_words)}
        self.vocab_size = len(self.word_index) + 2 # +2 for pad and OOV

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            cleaned = clean_text(text)
            words = cleaned.split()
            seq = [self.word_index.get(w, 1) for w in words] # 1 for OOV
            
            # Padding/Truncating to max_len=50 [cite: 31]
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[:self.max_len]
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.long)