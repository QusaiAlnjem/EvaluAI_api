import torch

def convert_predictions_to_aspects(text, predictions, offset_mapping, input_ids, tokenizer):
  aspects = []
  current_aspect = None
  
  for i, (pred_label, (start_pos, end_pos)) in enumerate(zip(predictions, offset_mapping)):
    # Skip special tokens and padding
    if start_pos == 0 and end_pos == 0:
      continue
        
    token = tokenizer.decode([input_ids[i]], skip_special_tokens=True)
    
    if pred_label == 1:  # B- (Beginning of aspect)
      # Save previous aspect if exists
      if current_aspect is not None:
        aspects.append(current_aspect)
      
      # Start new aspect
      current_aspect = {
        "term": text[start_pos:end_pos],
        "from": start_pos,
        "to": end_pos,
        "tokens": [token]
      }
        
    elif pred_label == 2:  # I- (Inside aspect)
      if current_aspect is not None:
        # Extend current aspect
        current_aspect["term"] = text[current_aspect["from"]:end_pos]
        current_aspect["to"] = end_pos
        current_aspect["tokens"].append(token)
      
      else:
        current_aspect = {
          "term": text[start_pos:end_pos],
          "from": start_pos,
          "to": end_pos,
          "tokens": [token]
        }
        
    elif pred_label == 0:  # O (Outside)
      # End current aspect if exists
      if current_aspect is not None:
        aspects.append(current_aspect)
        current_aspect = None
  
  if current_aspect is not None:
    aspects.append(current_aspect)
  
  return aspects

def aspects_extraction(text, model, tokenizer, max_length=64):
  model.eval()
  encoding = tokenizer(
    text,
    truncation = True,
    padding = 'max_length',
    max_length = max_length,
    return_offsets_mapping = True,
    return_tensors = 'pt'
  )
  input_ids = encoding["input_ids"]
  attention_mask = encoding["attention_mask"]

  with torch.no_grad():
    outputs = model(input_ids = input_ids, attention_mask = attention_mask)
    predicted_labels = torch.argmax(outputs.logits, dim=-1) # outputs.logits shape: [1, 64, 3]

  aspects = convert_predictions_to_aspects(text,
                                           predicted_labels[0].cpu().numpy(),
                                           encoding["offset_mapping"][0].cpu().numpy(),
                                           input_ids[0].cpu().numpy(),
                                           tokenizer)

  return aspects

def classify_sentiment(text, aspect, model, tokenizer):
  inputs = tokenizer(
    text, aspect,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=128
  )
  with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
  label_map = {0: "negative", 1: "neutral", 2: "positive"}
  return label_map[prediction]

def classify_product(text, model, tokenizer, label_map = ['airpods', 'ipad', 'laptop', 'phone', 'watch'], max_length=32):
  """
  Predict the product class for a single input text.
  Returns the predicted product label and confidence score.
  """
  model.eval()
  inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=max_length
  )
  with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
  predicted_product = label_map[pred_idx]
  return predicted_product
