from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = BertTokenizer.from_pretrained("sentiment_tokenizer")

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def sentiment_analysis(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]

    results = [labels[pred] for pred in predictions]
    return results

test_texts = [
    "I love this product! It's amazing.",
    "The service was terrible, very disappointed."
    "It's okay, not the best but not the worst either."
]

sentiments = sentiment_analysis(test_texts)

for text, sentiment in zip(test_texts, sentiments):
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}\n")
