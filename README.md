
# Amazon Product Sentiment Analysis

This project scrapes product reviews from [Amazon India](https://www.amazon.in) and uses a fine-tuned BERT model to perform sentiment analysis (positive, neutral, negative) on them. Reviews are collected concurrently and processed in batch for efficient analysis.

## Features

- Scrapes product links and reviews using Selenium
- Trains and fine-tunes a BERT model on labeled tweet sentiment data
- Performs batch sentiment analysis on reviews
- Generates a sentiment summary per product

## Setup

### Prerequisites

- Python 3.8+
- Chrome browser
- [ChromeDriver](https://chromedriver.chromium.org/downloads) matching your Chrome version (add to PATH)

### Install dependencies

```bash
pip install transformers datasets scikit-learn pandas selenium torch
````

### Data

Place your training and testing data in the following format:

* `tweet_sentiment_extraction/train.jsonl`
* `tweet_sentiment_extraction/test.jsonl`

Each entry in the JSONL files should have a `text` and `label` field.

### Train the Model

Run the training script to fine-tune BERT:

```bash
python train_model.py
```

This saves:

* `sentiment_model/` – the fine-tuned model
* `sentiment_tokenizer/` – tokenizer used for predictions

### Run Sentiment Analysis

```bash
python analyze_sentiments.py
```

It will:

* Prompt Amazon India for the given product name
* Collect reviews for the top 20 results
* Analyze each review’s sentiment
* Print a report for each product with sentiment counts

### Example Output

```
Product 1 URL: https://www.amazon.in/...
Sentiment Counts:
  Positive: 13
  Neutral: 5
  Negative: 2
```

## Notes

* Uses headless Chrome via Selenium for scraping
* Utilizes `ThreadPoolExecutor` for concurrent page scraping
* Sentiment labels are: `positive`, `neutral`, `negative`

## License

MIT License

## Author

Srimadhav Seebu Kumar



