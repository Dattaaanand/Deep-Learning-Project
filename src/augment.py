import pandas as pd
import nltk
from nltk.corpus import wordnet
import random
import re

# Ensure WordNet is downloaded
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

def get_synonyms(word):
    """Get a set of synonyms for a word from WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            if candidate != word and len(candidate.split()) == 1:
                synonyms.add(candidate)
    return list(synonyms)

def augment_text(text, num_replacements=1):
    """Replace random words in the text with synonyms."""
    words = text.split()
    if len(words) < 3: # Skip very short tweets
        return text
    
    # Find words that have synonyms
    augmentable_indices = [i for i, w in enumerate(words) if len(get_synonyms(w)) > 0]
    
    if not augmentable_indices:
        return text
    
    # Choose random words to replace
    num_to_replace = min(len(augmentable_indices), num_replacements)
    indices_to_replace = random.sample(augmentable_indices, num_to_replace)
    
    new_words = words[:]
    for idx in indices_to_replace:
        synonyms = get_synonyms(words[idx])
        if synonyms:
            new_words[idx] = random.choice(synonyms)
            
    return " ".join(new_words)

def run_augmentation():
    print("Augmenting Training Data...")
    
    # 1. Load existing training data
    try:
        df = pd.read_csv('../data/train.csv')
    except FileNotFoundError:
        print("Error: '../data/train.csv' not found.")
        return

    original_len = len(df)
    print(f"Original size: {original_len} tweets")

    # 2. Generate Augmentations
    # We will create ONE augmented version for every tweet
    augmented_rows = []
    
    for _, row in df.iterrows():
        # Keep original
        augmented_rows.append(row)
        
        # Create augmented version
        new_text = augment_text(row['text'])
        if new_text != row['text']:
            new_row = row.copy()
            new_row['text'] = new_text
            augmented_rows.append(new_row)

    # 3. Save new dataset
    df_augmented = pd.DataFrame(augmented_rows)
    
    # Shuffle the dataset so original and augmented aren't always next to each other
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path = '../data/train_augmented.csv'
    df_augmented.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"Augmentation Complete!")
    print(f"New size: {len(df_augmented)} tweets (Approx {len(df_augmented)/original_len:.1f}x larger)")
    print(f"Saved to: {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    run_augmentation()