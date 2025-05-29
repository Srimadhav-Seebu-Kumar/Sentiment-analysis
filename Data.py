import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

train_dataset = load_dataset('json', data_files=r'tweet_sentiment_extraction\train.jsonl', split='train')
test_dataset = load_dataset('json', data_files=r'tweet_sentiment_extraction\test.jsonl', split='train')

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)  

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",  
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print(results)

model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_tokenizer")
