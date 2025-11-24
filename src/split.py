import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    # 1. Load your original dataset
    try:
        df = pd.read_csv('../data/disaster_dataset.csv')
        print(f"Original Dataset Size: {len(df)}")
    except FileNotFoundError:
        print("Error: 'disaster_dataset.csv' not found. Please check the file path.")
        return

    # 2. Perform Stratified Split
    # test_size=2000 takes exactly 2000 rows for the test set
    # stratify=df['target'] ensures the class balance is preserved
    # random_state=42 ensures you get the same split every time you run this
    train_df, test_df = train_test_split(
        df, 
        test_size=2000, 
        stratify=df['target'], 
        random_state=42
    )

    # 3. Save the new Training Set (The remaining data)
    train_df.to_csv('../data/train.csv', index=False)
    
    # 4. Save the Test Labels (ID and Target only)
    # This is your "Answer Key"
    test_label = test_df[['id', 'target']]
    test_label.to_csv('../data/test_label.csv', index=False)
    
    # 5. Save the Test File (Inputs only, drop the target)
    # This is what you feed into the model
    test_file = test_df.drop(columns=['target'])
    test_file.to_csv('../data/test.csv', index=False)

    # Summary Output
    print("-" * 30)
    print("Files Created Successfully:")
    print(f"1. train.csv  ({len(train_df)} rows) - Use this for training")
    print(f"2. test.csv       ({len(test_file)} rows) - Use this for prediction")
    print(f"3. test_label.csv ({len(test_label)} rows) - Use this for evaluation")
    print("-" * 30)

if __name__ == "__main__":
    split_dataset()