import streamlit as st
import pandas as pd

# Streamlit App Title
st.title("Stock Sentiment Analysis & Prediction")

# Create a form for user input
with st.form(key="stock_form"):
    st.subheader("Enter Stock Information")
    
    # User inputs for stock name and code
    news_search = st.text_input("Stock Name (e.g., KLCI, Apple, Tesla)", value="KLCI")
    stock_code = st.text_input("Stock Code (e.g., ^KLSE, AAPL, TSLA)", value="^KLSE")
    
    # Submit button
    submit_button = st.form_submit_button(label="Submit")

# Process user input after submission
if submit_button:
    st.success(f"You have selected: **{news_search}** with stock code **{stock_code}**")
    
    # You can now pass `news_search` and `stock_code` into your scraping and prediction functions.
    st.write("Proceeding with data scraping and sentiment analysis...")

    #############################################Scraping##################################################
    duration_days = 30
    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime, timedelta
    import pytz  # For handling timezones
    import os  # For handling file operations
    import csv  # For writing to CSV
    import urllib.parse
    
    # Get today's date in the required format (YYYY-MM-DD)
    today = datetime.today().strftime("%Y-%m-%d")
    five_days_ago = (datetime.today() - timedelta(days=duration_days)).strftime("%Y-%m-%d")
    
    # Function to get user input and construct search URL
    def get_search_url():
        stock_name = news_search
        encoded_stock = urllib.parse.quote(stock_name)  # Encode stock name for URL
        base_url = f"https://theedgemalaysia.com/news-search-results?keywords={encoded_stock}&to={today}&from={five_days_ago}&language=english&offset="
        return base_url, stock_name  # Return both URL and stock name for file naming
    
    # Get search URL from user input
    base_url, stock_name = get_search_url()
    
    # CSV file path
    csv_file = "scraped_articles.csv"
    
    # Function to initialize the CSV file with a header
    def initialize_csv(file_path):
        if not os.path.exists(file_path):  # Only create the file if it does not exist
            with open(file_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Date", "title", "detail", "combined_text"])  # Column headers
    
    # Function to scrape a single page
    def scrape_page(page_number):
        # Construct the page URL
        url = f"{base_url}{page_number}"
        print(f"Scraping page: {url}")
    
        # Send an HTTP GET request
        response = requests.get(url)
    
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.content, "html.parser")
    
            # Find the containers with the headline, description, and timestamp
            articles = soup.find_all("div", class_="col-md-8 col-12")
    
            scraped_data = []  # Store extracted data before writing
    
            for article in articles:
                # Extract the headline
                headline = article.find("span", class_="NewsList_newsListItemHead__dg7eK")
                headline_text = headline.get_text(strip=True).replace(":", ".") if headline else "No headline found"
    
                # Extract the description
                description = article.find("span", class_="NewsList_newsList__2fXyv")
                description_text = description.get_text(strip=True).replace(":", ".") if description else "No description found"
    
                # Combine title and detail into one column
                combined_text = f"{headline_text} - {description_text}"
    
                # Extract the timestamp
                timestamp_container = article.find_next("div", class_="NewsList_infoNewsListSub__Ui2_Z")
                timestamp_text = timestamp_container.get_text(strip=True) if timestamp_container else "No timestamp found"
    
                # Clean the timestamp text to remove "(Updated)" or similar annotations
                if timestamp_text != "No timestamp found":
                    cleaned_timestamp_text = timestamp_text.split("(")[0].strip()  # Remove "(Updated)" or any text in parentheses
    
                    try:
                        # Parse the cleaned timestamp text (e.g., "26 Dec 2024, 05:57 pm")
                        datetime_obj = datetime.strptime(cleaned_timestamp_text, "%d %b %Y, %I:%M %p")
    
                        # Add timezone
                        timezone = pytz.timezone("Asia/Kuala_Lumpur")  # Replace with desired timezone
                        localized_datetime = timezone.localize(datetime_obj)
    
                        # Convert to ISO 8601 format
                        iso_timestamp = localized_datetime.isoformat()
                    except ValueError:
                        iso_timestamp = "Invalid timestamp format"
                else:
                    iso_timestamp = "No timestamp available"
    
                # Store data in a list
                scraped_data.append([iso_timestamp, headline_text, description_text, combined_text])
    
            # Write data to the CSV file
            with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows(scraped_data)
    
            print(f"Saved {len(scraped_data)} articles to CSV.")
            print("-" * 50)
        else:
            print(f"Failed to retrieve page {page_number}. Status code: {response.status_code}")
    
    # Initialize CSV file
    initialize_csv(csv_file)
    
    # Scrape all pages (maximum 10 pages)
    for page in range(0, 100, 10):  # Adjust the range based on your needs
        scrape_page(page) 

    #############################Sentiment Analysis################################
    # Load the dataset
    file_path = 'scraped_articles.csv'
    data = pd.read_csv(file_path)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    # Load the saved model and tokenizer
    model_path = "StephanAkkerman/FinTwitBERT-sentiment"  
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def analyze_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)  # Convert logits to probabilities
        
        # Extract probabilities for each sentiment
        positive = probabilities[0][1].item()
        neutral = probabilities[0][0].item()
        negative = probabilities[0][2].item()
        
        # Calculate custom sentiment score
        score = positive - negative
        
        return positive, neutral, negative, score
        
    # Assuming the text data is in the 'content' column
    data['positive'], data['neutral'], data['negative'], data['score'] = zip(*data['combined_text'].apply(analyze_sentiment))
    # Add entry count for each row
    data['entry_count'] = 1
    sentiment_data_path = "sentiment_analysis_results.csv"
    data.to_csv(sentiment_data_path, index=False)

    # Streamlit App Title
    st.subheader("📊 Sentiment Scores of Latest 5 News Articles")
    
    # Sort by date to get the latest news
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.sort_values(by='Date', ascending=False).head(5)
    
    # Display the first 5 news articles along with their sentiment scores
    for i, row in data.iterrows():
        st.markdown(f"### {i+1}. {row['title']}")
        st.write(f"📅 **Date:** {row['Date'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"📰 **Summary:** {row['detail']}")
        
        # Display Sentiment Scores
        st.write(f"📈 **Positive Sentiment:** {row['positive']:.2f}")
        st.write(f"⚖ **Neutral Sentiment:** {row['neutral']:.2f}")
        st.write(f"📉 **Negative Sentiment:** {row['negative']:.2f}")
        st.write(f"🧮 **Overall Sentiment Score:** {row['score']:.2f}")
        st.write("---")  # Divider for clarity

        
#st.write(f"👍 **Positive:** `{row['positive']:.2f}` | 😐 **Neutral:** `{row['neutral']:.2f}` | 👎 **Negative:** `{row['negative']:.2f}`")
