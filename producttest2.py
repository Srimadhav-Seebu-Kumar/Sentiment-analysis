import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained BERT model and tokenizer for sentiment analysis
model = BertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = BertTokenizer.from_pretrained("sentiment_tokenizer")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Sentiment analysis function
def sentiment_analysis(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    results = [labels[pred] for pred in predictions]
    return results

# Set up Selenium WebDriver
driver = webdriver.Chrome()  # Make sure chromedriver is in your PATH
driver.implicitly_wait(10)

# Function to fetch top 20 products from Amazon's bestseller page
def get_top_20_products():
    driver.get("https://www.amazon.in/gp/bestsellers")
    time.sleep(2)  # Wait for page to load
    products = []
    for i in range(1, 21):
        try:
            product_element = driver.find_element(By.XPATH, f"(//div[@id='gridItemRoot'])[{i}]//a[@class='a-link-normal']")
            product_name = product_element.text
            product_url = product_element.get_attribute("href")
            products.append((product_name, product_url))
        except Exception as e:
            print(f"Error retrieving product {i}: {e}")
    return products

# Function to fetch top 20 reviews for a given product URL
def get_top_20_reviews(product_url):
    driver.get(product_url)
    time.sleep(2)  # Wait for page to load
    reviews = []
    try:
        # Navigate to the reviews section
        reviews_link = driver.find_element(By.XPATH, "//a[@data-hook='see-all-reviews-link-foot']")
        reviews_link.click()
        time.sleep(2)
        # Extract the top 20 reviews
        review_elements = driver.find_elements(By.XPATH, "//div[@data-hook='review-collapsed']")[:20]
        for review in review_elements:
            reviews.append(review.text)
    except Exception as e:
        print(f"Error retrieving reviews: {e}")
    return reviews

# Main function to analyze sentiments for each of the top 20 products
def analyze_top_20_product_sentiments():
    products = get_top_20_products()
    results = []
    for product_name, product_url in products:
        reviews = get_top_20_reviews(product_url)
        if not reviews:
            continue
        sentiments = sentiment_analysis(reviews)
        sentiment_counts = {
            "positive": sentiments.count("positive"),
            "neutral": sentiments.count("neutral"),
            "negative": sentiments.count("negative")
        }
        results.append({
            "product_name": product_name,
            "url": product_url,
            "sentiment_counts": sentiment_counts
        })
    
    # Display the results
    for product in results:
        print(f"Product Name: {product['product_name']}")
        print(f"Product URL: {product['url']}")
        print("Sentiment Counts:")
        for sentiment, count in product["sentiment_counts"].items():
            print(f"  {sentiment.capitalize()}: {count}")
        print("\n")

    # Close the browser
    driver.quit()

# Run the main function
if __name__ == "__main__":
    analyze_top_20_product_sentiments()
