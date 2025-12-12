import os
import sys
import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.preprocess import preprocess_dataframe
from src.models.distilbert_model import FakeNewsClassifier



def load_kaggle_dataset(raw_dir="data/raw"):

    print("📂 Loading Kaggle Fake/True dataset...")

    possible_fake = ["Fake.xlsx", "Fake.xls", "Fake.csv"]
    possible_true = ["True.xlsx", "True.xls", "True.csv"]

    fake_path = next((os.path.join(raw_dir, f) for f in possible_fake if os.path.exists(os.path.join(raw_dir, f))), None)
    true_path = next((os.path.join(raw_dir, f) for f in possible_true if os.path.exists(os.path.join(raw_dir, f))), None)

    if not fake_path or not true_path:
        raise FileNotFoundError("❌ Fake or True dataset missing in data/raw")

    if fake_path.endswith(".csv"):
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
    else:
        df_fake = pd.read_excel(fake_path)
        df_true = pd.read_excel(true_path)

    df_fake["label"] = "FAKE"
    df_true["label"] = "REAL"

    df = pd.concat([df_fake, df_true], ignore_index=True)

    if "text" not in df.columns:
        raise KeyError("❌ Kaggle dataset missing 'text' column!")

    return df[["text", "label"]]



def load_liar_dataset(raw_dir="data/raw"):

    print("📂 Loading LIAR dataset for cross-evaluation...")

    test_path = os.path.join(raw_dir, "test.tsv")
    if not os.path.exists(test_path):
        raise FileNotFoundError("❌ test.tsv not found for LIAR dataset")

    df = pd.read_csv(test_path, sep="\t", header=None)
    df = df.iloc[:, [1, 2]]  
    df.columns = ["label", "text"]

    def map_liar_label(lbl):
        fake = ["false", "pants-fire"]
        real = ["half-true", "mostly-true", "true"]
        return 1 if lbl in fake else 0 if lbl in real else None

    df["label"] = df["label"].apply(map_liar_label)
    df = df.dropna()

    return preprocess_dataframe(df, "text", "label")



class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length
        )
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item



def evaluate(model, loader, device, name="Dataset"):
    model.eval()
    preds, truth = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred_class = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred_class)
            truth.extend(batch["labels"].cpu().numpy())

    print(f"\n📊 Evaluation on {name}")
    print(f"Accuracy: {accuracy_score(truth, preds):.4f}")
    print(f"F1-score: {f1_score(truth, preds):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(truth, preds))
    print("Classification Report:\n", classification_report(truth, preds))



def train_model(epochs=2, batch_size=8, lr=2e-5, use_gpu=True):

    kaggle_df = load_kaggle_dataset()
    kaggle_df = preprocess_dataframe(kaggle_df, "text", "label")
    kaggle_df["label"] = kaggle_df["label"].map({"REAL": 0, "FAKE": 1})

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        kaggle_df["text"].tolist(),
        kaggle_df["label"].tolist(),
        test_size=0.2,
        random_state=42
    )

    classifier = FakeNewsClassifier(num_labels=2)
    train_ds = NewsDataset(train_texts, train_labels, classifier.tokenizer)
    val_ds = NewsDataset(val_texts, val_labels, classifier.tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    classifier.model.to(device)
    print(f"\n🚀 Training on: {device}")

    optimizer = AdamW(classifier.model.parameters(), lr=lr)

    
    for epoch in range(epochs):
        classifier.model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = classifier.model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"\n📌 Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(train_loader):.4f}")
        evaluate(classifier.model, val_loader, device, "Kaggle Validation")

   
    classifier.model.save_pretrained("models/checkpoints")
    classifier.tokenizer.save_pretrained("models/tokenizer")

    print("\n📥 Testing generalization on LIAR dataset...")
    liar_df = load_liar_dataset()
    liar_ds = NewsDataset(liar_df["text"].tolist(), liar_df["label"].tolist(), classifier.tokenizer)
    liar_loader = DataLoader(liar_ds, batch_size=batch_size)

    evaluate(classifier.model, liar_loader, device, "LIAR Test")

    print("\n🎯 Training + Evaluation Completed Successfully!")
    return classifier


if __name__ == "__main__":
    train_model()
