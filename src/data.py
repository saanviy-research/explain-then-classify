"""Download and load the LonXplain dataset."""
import urllib.request

import pandas as pd

import config


def load_dataset() -> pd.DataFrame:
    """Download the dataset if needed, load it, and normalize column names."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not config.DATASET_PATH.exists():
        print("Downloading LonXplain dataset...")
        urllib.request.urlretrieve(config.DATASET_URL, config.DATASET_PATH)
    else:
        print("Dataset already exists.")

    df = pd.read_csv(config.DATASET_PATH)
    print(f"Raw columns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")

    df = df.rename(columns={"tb": "label", "tb_exp": "explanation"})
    df["label"] = df["label"].astype(int)

    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"Present rate: {df['label'].mean():.2%}")

    return df
