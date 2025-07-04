import torch
import pickle
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification, BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer

tokenizer_sent = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer_asp = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer_cls = AutoTokenizer.from_pretrained("roberta-base")

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

def classify_product(text, model, tokenizer, label_encoder, max_length=32):
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
    predicted_product = label_encoder.inverse_transform([pred_idx])[0]
    return predicted_product

""" REMOVE THE FUNCTION AFTER APPLYING CHANGES """
def format_output(text, aspects, sentiments, product):
  print(f"Text: {text}")
  print(f"Found {len(aspects)} aspects:")
  
  for i, aspect in enumerate(aspects, 1):
    print(f"  {i}. [{product}] '{aspect['term']}' -> {sentiments[i-1]}")
      
  print("-" * 50)

if __name__ == "__main__":
  AspectModel = BertForTokenClassification.from_pretrained("aspect_extraction_model\checkpoint-12207")
  SentimentModel = BertForSequenceClassification.from_pretrained("absa_model/checkpoint-10456")
  ClassificationModel = AutoModelForSequenceClassification.from_pretrained("product_classifier")
  with open(r"product_classifier\label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

  """ THIS PART SHOULD BE CHANGED """
  test_data = pd.read_csv(r"C:\Users\HP\Downloads\testing and evaluation\Restaurants_Test_Data_PhaseA.csv")
  test_data = test_data.tail(5)
  for _ , row in test_data.iterrows():
    aspects = aspects_extraction(row['Sentence'], AspectModel, tokenizer_asp)
    product = classify_product(row["Sentence"], ClassificationModel, tokenizer_cls, label_encoder)
    sentiments = []
    for _, aspect in enumerate(aspects):
      s = classify_sentiment(row['Sentence'], aspect['term'], SentimentModel, tokenizer_sent)
      sentiments.append(s)
    format_output(row['Sentence'], aspects, sentiments, product)