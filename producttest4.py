from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
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

# Configure headless browser settings
chrome_options = Options()
chrome_options.add_argument("--headless")

def fetch_product_links_amazon(product_name):
    # Use a temporary browser instance to search for product links
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.amazon.in")
    
    # Search for the product
    search_box = driver.find_element(By.ID, "twotabsearchtextbox")
    search_box.send_keys(product_name)
    search_box.send_keys(Keys.RETURN)
    time.sleep(2)  # Allow search results to load
    
    # Collect the first 20 product links
    product_links = [product.get_attribute("href") for product in driver.find_elements(By.XPATH, "//h2/a")[:20]]
    driver.quit()
    return product_links

def fetch_reviews_for_product(product_link):
    # Create a new browser instance for each product link
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(product_link)
    
    # Scroll down and open the reviews section
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    
    # Collect the first 20 reviews
    reviews = [review.text for review in driver.find_elements(By.XPATH, "//div[@data-hook='review-collapsed']")[:20]]
    driver.quit()  # Close this browser instance
    return reviews

def fetch_all_reviews_amazon(product_name):
    # Fetch product links
    product_links = fetch_product_links_amazon(product_name)
    
    # Dictionary to store reviews for each product
    product_reviews = {}
    
    # Use ThreadPoolExecutor to fetch reviews concurrently with multiple browser instances
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_reviews_for_product, product_links)
    
    # Collect reviews for each product link
    for i, reviews in enumerate(results):
        product_reviews[f"Product {i+1}"] = {
            "product_link": product_links[i],
            "reviews": reviews
        }
    
    return product_reviews

def batch_sentiment_analysis(texts):
    # Perform sentiment analysis in batch
    if not texts:  # Handle empty input list
        return []
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    results = [labels[pred] for pred in predictions]
    return results

def analyze_product_sentiments(product_name):
    # Fetch all product reviews concurrently
    product_reviews = fetch_all_reviews_amazon(product_name)
    
    sentiment_summary = {}
    all_reviews = []
    product_indices = []
    
    # Gather all reviews to process in batch
    for product, details in product_reviews.items():
        url = details["product_link"]
        reviews = details["reviews"]
        all_reviews.extend(reviews)
        product_indices.extend([product] * len(reviews))  # Track product to review mapping
    
    # Batch process all reviews for sentiment analysis
    sentiments = batch_sentiment_analysis(all_reviews)
    
    # Count sentiments per product
    for product in set(product_indices):
        relevant_sentiments = [sentiments[i] for i in range(len(sentiments)) if product_indices[i] == product]
        sentiment_counts = {
            "positive": relevant_sentiments.count("positive"),
            "neutral": relevant_sentiments.count("neutral"),
            "negative": relevant_sentiments.count("negative")
        }
        sentiment_summary[product] = {
            "url": product_reviews[product]["product_link"],
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
