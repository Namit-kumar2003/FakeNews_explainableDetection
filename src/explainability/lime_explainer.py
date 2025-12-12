
import numpy as np
from lime.lime_text import LimeTextExplainer

def make_predict_proba_fn(model, tokenizer, device, max_length=64):
    
    def predict_proba(texts):
        model.eval()
        all_probs = []
        
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            with __no_grad_context():
                outputs = model(**enc)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    
    import torch
    def __no_grad_context():
        return torch.no_grad()
    return predict_proba


def get_lime_explanation(text, model, tokenizer, device=None, class_names=None, num_features=10, max_length=64):
    
    if device is None:
        device = next(model.parameters()).device

    if class_names is None:
        class_names = ["REAL", "FAKE"]

    predict_fn = make_predict_proba_fn(model, tokenizer, device, max_length=max_length)
    explainer = LimeTextExplainer(class_names=class_names)

    
    probs = predict_fn([text])[0]
    pred_class = int(probs.argmax())

    exp = explainer.explain_instance(text, predict_fn, num_features=num_features, labels=(pred_class,))
    
    word_importances = exp.as_list(label=pred_class)
    
    highlights = [{"word": w, "importance": float(score)} for w, score in word_importances]
    return pred_class, highlights
