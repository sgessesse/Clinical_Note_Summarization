import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import re

def clean_text(text):
    """Removes instructions before '###' and '### Response: ' suffix."""
    # Find the position of '###'
    marker_pos = text.find('###')
    if marker_pos != -1:
        # Get text after '###', strip leading/trailing whitespace
        cleaned_text = text[marker_pos + 3:].strip()
    else:
        # If '###' not found, use the original text (or decide on alternative handling)
        cleaned_text = text.strip()

    # Remove the suffix '### Response: ' if it exists at the end
    suffix = "### Response:" # Note: Adjusted to match common patterns, check exact spacing if needed
    if cleaned_text.endswith(suffix):
        cleaned_text = cleaned_text[:-len(suffix)].strip()
    # Also handle potential variations like just "Response:"
    elif cleaned_text.endswith("Response:"):
         cleaned_text = cleaned_text[:-len("Response:")].strip()

    return cleaned_text

def load_and_split_data(parquet_path="data/train-00000-of-00001.parquet", test_size=0.1, validation_size=0.1):
    """Loads data, cleans text, and splits into train, validation, test sets."""
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} records.")

    # Ensure columns exist
    if 'text' not in df.columns or 'summary' not in df.columns:
        raise ValueError("Parquet file must contain 'text' and 'summary' columns.")

    print("Cleaning 'text' column...")
    # Apply cleaning function - handle potential NaN values if any
    df['text'] = df['text'].fillna('').astype(str).apply(clean_text)
    df['summary'] = df['summary'].fillna('').astype(str) # Ensure summary is string

    print("Splitting data...")
    # Calculate split sizes
    train_val_size = 1.0 - test_size
    relative_validation_size = validation_size / train_val_size

    # Split into train+validation and test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42 # for reproducibility
    )

    # Split train+validation into train and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_validation_size,
        random_state=42 # use the same random state for consistency
    )

    print(f"Split complete: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Combine into a DatasetDict
    raw_datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    print("DatasetDict created.")
    print(raw_datasets)

    return raw_datasets

if __name__ == "__main__":
    # Example usage: Load and split the data
    raw_datasets = load_and_split_data()

    # You can access individual splits like this:
    # print("\nSample from training set:")
    # print(raw_datasets['train'][0])
