import re
import torch
import numpy as np
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK data (run once)
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
# We remove common "stop words" (the, is, at) to reduce noise
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. Standard Cleaning (from your PDF)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # URLs
    text = re.sub(r'@\w+', '', text) # Mentions
    text = re.sub(r'#(\w+)', r'\1', text) # Hashtags
    text = re.sub(r'[^a-zA-Z0-9\s\']', '', text) # Punctuation
    
    # 2. Tokenize
    words = text.split()
    
    # 3. Lemmatization & Stopword Removal (The Upgrade)
    # "running" -> "run", "cars" -> "car"
    # Removes "the", "a", "is" to focus on keywords
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    # Join back to string
    text = " ".join(cleaned_words)
    return text.strip()

class TextTokenizer:
    def __init__(self, max_words=10000, max_len=50):
        self.max_words = max_words
        self.max_len = max_len
        self.word_index = {}
        self.vocab_size = 0

    def fit(self, texts):
        all_words = []
        for text in texts:
            all_words.extend(clean_text(text).split())
        
        count = Counter(all_words)
        common_words = count.most_common(self.max_words - 2)
        
        self.word_index = {word: i + 2 for i, (word, _) in enumerate(common_words)}
        self.vocab_size = len(self.word_index) + 2

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            cleaned = clean_text(text)
            words = cleaned.split()
            seq = [self.word_index.get(w, 1) for w in words]
            
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[:self.max_len]
            sequences.append(seq)
        return torch.tensor(sequences, dtype=torch.long)