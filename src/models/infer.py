
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import os
import numpy as np

from src.explainability.attention_visualizer import get_attention_importances
from src.explainability.lime_explainer import get_lime_explanation, make_predict_proba_fn

LABELS = {0: "REAL", 1: "FAKE"}

class FakeNewsInference:
    def __init__(self, model_dir="models/checkpoints", tokenizer_dir="models/tokenizer", use_gpu=True):
        device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.device = torch.device(device)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"📌 Model loaded on: {self.device}")

        
        self.predict_proba_fn = make_predict_proba_fn(self.model, self.tokenizer, self.device, max_length=64)

    def predict(self, text):
        
        enc = self.tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class = int(probs.argmax())
        label = LABELS[pred_class]
        confidence = float(probs[pred_class])

        
        tokens, attn_scores = get_attention_importances(text, self.model, self.tokenizer, device=self.device, max_length=64)

       
        token_display = []
        for t in tokens:
          
            if t.startswith("##"):
                token_display.append(t[2:])
            else:
                token_display.append(t)

        attn_list = []
        for tok, score in zip(token_display, attn_scores):
            attn_list.append({"token": tok, "score": float(score)})

        
        pred_cls_for_lime, lime_highlights = get_lime_explanation(text, self.model, self.tokenizer, device=self.device, class_names=["REAL","FAKE"], num_features=10, max_length=64)

       
        
        merged = {}
        
        for item in attn_list:
            key = item["token"].lower()
            merged.setdefault(key, {"token": item["token"], "attn": 0.0, "lime": 0.0})
            merged[key]["attn"] = max(merged[key]["attn"], float(item["score"]))

        
        for w in lime_highlights:
            key = w["word"].lower()
            merged.setdefault(key, {"token": w["word"], "attn": 0.0, "lime": 0.0})
            merged[key]["lime"] = max(merged[key]["lime"], float(w["importance"]))

       
        attn_vals = np.array([v["attn"] for v in merged.values()]) if merged else np.array([0.0])
        lime_vals = np.array([v["lime"] for v in merged.values()]) if merged else np.array([0.0])

        def normalize(arr):
            arr = np.array(arr, dtype=float)
            if arr.max() - arr.min() > 0:
                return (arr - arr.min()) / (arr.max() - arr.min())
            return arr - arr.min()

        attn_norm = normalize(attn_vals)
        lime_norm = normalize(lime_vals)

        merged_list = []
        for i, (k, v) in enumerate(merged.items()):
            combined_score = 0.6 * attn_norm[i] + 0.4 * lime_norm[i] 
            merged_list.append({
                "token_or_word": v["token"],
                "attention_score": float(attn_norm[i]),
                "lime_score": float(lime_norm[i]),
                "combined_score": float(combined_score)
            })

        
        merged_list = sorted(merged_list, key=lambda x: x["combined_score"], reverse=True)

        return {
            "text": text,
            "prediction": label,
            "confidence": round(confidence, 4),
            "attention_tokens": attn_list,
            "lime_highlights": lime_highlights,
            "merged_highlights": merged_list
        }


if __name__ == "__main__":
    
    inf = FakeNewsInference()
    while True:
        txt = input("\nEnter text (or 'exit'): ")
        if txt.strip().lower() == "exit":
            break
        out = inf.predict(txt)
        print("\nPrediction:", out["prediction"], "Confidence:", out["confidence"])
        print("\nTop merged highlights:")
        for h in out["merged_highlights"][:10]:
            print(f"{h['token_or_word']}: combined={h['combined_score']:.3f} (attn={h['attention_score']:.3f}, lime={h['lime_score']:.3f})")
