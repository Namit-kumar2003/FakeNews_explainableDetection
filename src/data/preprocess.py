import re
import string

def clean_text(text: str) -> str:
    # Basic text cleaning: lowercase, remove punctuation, extra spaces
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)  # remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataframe(df, text_column: str, label_column: str = None):
    df = df.copy()
    df[text_column] = df[text_column].fillna('').apply(clean_text)
    if label_column is not None:
        df = df.dropna(subset=[label_column])
    return df
