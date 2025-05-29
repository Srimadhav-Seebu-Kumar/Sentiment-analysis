import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the sentiment model
model = BertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = BertTokenizer.from_pretrained("sentiment_tokenizer")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to perform sentiment analysis
def sentiment_analysis(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    results = [labels[pred] for pred in predictions]
    return results

# Set up Selenium WebDriver
driver = webdriver.Chrome()  # or use webdriver.Firefox() if using Firefox
driver.implicitly_wait(10)

def fetch_reviews_amazon(product_name):
    # Search on Amazon
    search_url = "https://www.amazon.in"
    driver.get(search_url)
    search_box = driver.find_element(By.ID, "twotabsearchtextbox")
    search_box.send_keys(product_name)
    search_box.send_keys(Keys.RETURN)
    time.sleep(2)  # wait for search results to load
    
    # Find the first product link and navigate to its page
    product_link = driver.find_element(By.XPATH, "//h2/a").get_attribute("href")
    driver.get(product_link)
    
    # Scroll down and open the reviews section
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    
    # Collect the first 20 reviews
    reviews = []
    review_elements = driver.find_elements(By.XPATH, "//div[@data-hook='review-collapsed']")[:20]
    for review in review_elements:
        reviews.append(review.text)
        
    return product_name, product_link, reviews

def analyze_product_sentiments(product_name):
    # Get product name, URL, and top 20 reviews
    name, url, reviews = fetch_reviews_amazon(product_name)
    
    # Run sentiment analysis on the reviews
    sentiments = sentiment_analysis(reviews)
    
    # Count the sentiment types
    sentiment_counts = {
        "positive": sentiments.count("positive"),
        "neutral": sentiments.count("neutral"),
        "negative": sentiments.count("negative")
    }
    
    return {
        "product_name": name,
        "url": url,
        "sentiment_counts": sentiment_counts
    }

# Example usage
product_name = "Harry Potter Book 1"
report = analyze_product_sentiments(product_name)

# Display the results
print(f"Product Name: {report['product_name']}")
print(f"Product URL: {report['url']}")
print("Sentiment Counts:")
for sentiment, count in report["sentiment_counts"].items():
    print(f"  {sentiment.capitalize()}: {count}")
