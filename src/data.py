"""Load, clean, and split the ticket data. Both models use this so the split is identical."""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_split(path, text_col, label_col, test_size=0.2, seed=42, min_class_count=10):
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]

    # drop exact duplicate tickets BEFORE splitting, otherwise the same ticket
    # can land in train and test and inflate the scores
    df = df.drop_duplicates(subset=text_col)

    # drop classes too small to split or evaluate
    counts = df[label_col].value_counts()
    df = df[df[label_col].isin(counts[counts >= min_class_count].index)]

    train, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
