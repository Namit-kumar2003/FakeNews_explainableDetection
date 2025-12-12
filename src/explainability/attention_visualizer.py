
import torch
import numpy as np

def get_attention_importances(text, model, tokenizer, device=None, max_length=64, top_k=10):
    
    if device is None:
        device = next(model.parameters()).device

    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True
    ).to(device)

    
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding, output_attentions=True)

    
    attentions = outputs.attentions  
   
    last_attn = attentions[-1].squeeze(0)  
    
    avg_last = last_attn.mean(dim=0).cpu().numpy() 

    
    cls_weights = avg_last[0]  
    
    attn_mask = encoding['attention_mask'].squeeze(0).cpu().numpy()  
    cls_weights = cls_weights * attn_mask  

    
    token_ids = encoding['input_ids'].squeeze(0).cpu().numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    
    
    valid_indices = [i for i, m in enumerate(attn_mask) if m == 1]
    valid_scores = cls_weights[valid_indices].copy()
    if valid_scores.max() - valid_scores.min() > 0:
        norm = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min())
    else:
        norm = valid_scores - valid_scores.min()

    
    scores = np.zeros_like(cls_weights)
    for idx, v in zip(valid_indices, norm):
        scores[idx] = float(v)

    
    return tokens, scores.tolist()
