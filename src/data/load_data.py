import os
import pandas as pd

def load_kaggle_fake_real(data_dir: str):
    """
    Load the Kaggle Fake/Real News dataset.
    Expects a CSV file (e.g. 'fake_and_real_news.csv') in data_dir.
    Returns pandas DataFrame.
    """
    path = os.path.join(data_dir, 'fake_and_real_news.csv')
    df = pd.read_csv(path)
    return df

def load_liar(data_dir: str):
    """
    Load the LIAR dataset (tsv files).
    Expects files like 'train.tsv', 'test.tsv', 'valid.tsv' under data_dir.
    Returns combined pandas DataFrame (train + test + valid).
    """
    parts = []
    for split in ['train', 'test', 'valid']:
        fname = os.path.join(data_dir, f'{split}.tsv')
        if os.path.exists(fname):
            df = pd.read_csv(fname, sep='\t', header=None, low_memory=False)
            parts.append(df)
    if parts:
        full = pd.concat(parts, ignore_index=True)
        return full
    else:
        raise FileNotFoundError("No LIAR data files found in " + data_dir)
