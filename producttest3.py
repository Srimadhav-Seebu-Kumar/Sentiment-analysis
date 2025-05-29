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
    # Initialize the driver (assuming it's set up earlier)
    search_url = "https://www.amazon.in"
    driver.get(search_url)
    
    # Search for the product
    search_box = driver.find_element(By.ID, "twotabsearchtextbox")
    search_box.send_keys(product_name)
    search_box.send_keys(Keys.RETURN)
    time.sleep(2)  # Wait for search results to load
    
    # Find the first 20 product links
    product_links = []
    product_elements = driver.find_elements(By.XPATH, "//h2/a")[:20]
    for product in product_elements:
        product_links.append(product.get_attribute("href"))
    
    # Dictionary to store reviews for each product
    product_reviews = {}
    
    # Loop through each product link and fetch reviews
    for i, product_link in enumerate(product_links, 1):
        driver.get(product_link)
        time.sleep(2)  # Allow page to load
        
        # Scroll down and open the reviews section
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Collect the first 20 reviews for each product
        reviews = []
        review_elements = driver.find_elements(By.XPATH, "//div[@data-hook='review-collapsed']")[:20]
        for review in review_elements:
            reviews.append(review.text)
        
        # Save reviews with product link
        product_reviews[f"Product {i}"] = {
            "product_link": product_link,
            "reviews": reviews
        }
        
    return product_reviews


def analyze_product_sentiments(product_name):
    # Get the dictionary of product reviews
    product_reviews = fetch_reviews_amazon(product_name)
    
    sentiment_summary = {}
    
    for product, details in product_reviews.items():
        url = details["product_link"]
        reviews = details["reviews"]
        
        # Check if there are reviews to analyze
        if not reviews:
            print(f"No reviews found for {product}. Skipping sentiment analysis.")
            sentiment_summary[product] = {
                "url": url,
                "sentiment_counts": {"positive": 0, "neutral": 0, "negative": 0}
            }
            continue
        
        # Run sentiment analysis on the reviews
        sentiments = sentiment_analysis(reviews)
        
        # Count the sentiment types
        sentiment_counts = {
            "positive": sentiments.count("positive"),
            "neutral": sentiments.count("neutral"),
            "negative": sentiments.count("negative")
        }
        
        # Add the sentiment counts for each product
        sentiment_summary[product] = {
            "url": url,
            "sentiment_counts": sentiment_counts
        }
    
    return sentiment_summary

# Example usage
product_name = "Harry Potter Book 1"
report = analyze_product_sentiments(product_name)

# Display the results
for product, details in report.items():
    print(f"{product} URL: {details['url']}")
    print("Sentiment Counts:")
    for sentiment, count in details["sentiment_counts"].items():
        print(f"  {sentiment.capitalize()}: {count}")


# Example usage
product_name = "Harry Potter Book 1"
report = analyze_product_sentiments(product_name)

# Display the results
for product, details in report.items():
    print(f"{product} URL: {details['url']}")
    print("Sentiment Counts:")
    for sentiment, count in details["sentiment_counts"].items():
        print(f"  {sentiment.capitalize()}: {count}")
