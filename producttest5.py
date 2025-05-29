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
    
    # Collect the first 20 products' links, names, and prices
    product_elements = driver.find_elements(By.XPATH, "//div[@data-component-type='s-search-result']")[:5]
    
    products = []
    for product in product_elements:
        # Get product link
        link_element = product.find_element(By.XPATH, ".//h2/a")
        product_link = link_element.get_attribute("href")
        
        # Get product name
        product_name_text = link_element.text
        
        # Get product price
        try:
            price_whole = product.find_element(By.XPATH, ".//span[@class='a-price-whole']").text
            price_fraction = product.find_element(By.XPATH, ".//span[@class='a-price-fraction']").text
            product_price = price_whole + price_fraction
        except:
            product_price = "Price not available"
        
        products.append({
            "product_link": product_link,
            "product_name": product_name_text,
            "product_price": product_price
        })
    
    driver.quit()
    return products

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
    products = fetch_product_links_amazon(product_name)
    
    product_reviews = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_reviews_for_product, [product['product_link'] for product in products])
    
    for i, reviews in enumerate(results):
        product_info = products[i]
        product_reviews[f"Product {i+1}"] = {
            "product_link": product_info['product_link'],
            "product_name": product_info['product_name'],
            "product_price": product_info['product_price'],
            "reviews": reviews
        }
    
    return product_reviews

def batch_sentiment_analysis(texts):
    # Perform sentiment analysis in batch
    if not texts:  
        return []
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    results = [labels[pred] for pred in predictions]
    return results

def analyze_product_sentiments(product_name):
    product_reviews = fetch_all_reviews_amazon(product_name)
    
    sentiment_summary = {}
    all_reviews = []
    product_indices = []
    
    for product_key, details in product_reviews.items():
        reviews = details["reviews"]
        all_reviews.extend(reviews)
        product_indices.extend([product_key] * len(reviews))  
    
    sentiments = batch_sentiment_analysis(all_reviews)
    
    for product_key in set(product_indices):
        relevant_sentiments = [sentiments[i] for i in range(len(sentiments)) if product_indices[i] == product_key]
        sentiment_counts = {
            "positive": relevant_sentiments.count("positive"),
            "neutral": relevant_sentiments.count("neutral"),
            "negative": relevant_sentiments.count("negative")
        }
        sentiment_summary[product_key] = {
            "url": product_reviews[product_key]["product_link"],
            "product_name": product_reviews[product_key]["product_name"],
            "product_price": product_reviews[product_key]["product_price"],
            "sentiment_counts": sentiment_counts
        }
    
    return sentiment_summary

product_name = "LEGO FLOWER BOUQUET"
report = analyze_product_sentiments(product_name)

for product, details in report.items():
    print(f"{product} Name: {details['product_name']}")
    print(f"Price: {details['product_price']}")
    print(f"URL: {details['url']}")
    print("Sentiment Counts:")
    for sentiment, count in details["sentiment_counts"].items():
        print(f"  {sentiment.capitalize()}: {count}")
    print()
