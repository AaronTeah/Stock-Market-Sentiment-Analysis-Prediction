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
    duration_days = 5
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
    for page in range(0, 50, 10):  # Adjust the range based on your needs
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
    st.title("📰 Latest 5 News Articles with Sentiment Scores")
    
    # File path of the uploaded dataset
    csv_file_path = "sentiment_analysis_results.csv"
    
    # Load the dataset
    try:
        df = pd.read_csv(csv_file_path)
    
        # Convert 'Date' column to datetime format for sorting
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
        # Remove duplicate articles based on 'title' and 'detail'
        df = df.drop_duplicates(subset=['title', 'detail'], keep='first')
    
        # Sort by date (latest first) and get the top 5 unique news
        latest_news = df.sort_values(by='Date', ascending=False).head(3)
    
        # Display the latest 5 unique news articles with sentiment scores
        for i, row in latest_news.iterrows():
            st.markdown(f"### {i+1}. {row['title']}")
            st.write(f"📅 **Date:** {row['Date'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"📰 **Summary:** {row['detail']}")
            
            # Show sentiment scores
            st.write(f"👍 **Positive:** `{row['positive']:.5f}` | 😐 **Neutral:** `{row['neutral']:.5f}` | 👎 **Negative:** `{row['negative']:.5f}`")
            st.write(f"🧮 **Overall Sentiment Score:** {row['score']:.5f}")
            st.write("---")  # Divider for clarity
    
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")

    # Load the CSV file
    file_path = "sentiment_analysis_results.csv" 
    df = pd.read_csv(file_path)
    
    # Remove unnecessary columns
    columns_to_remove = ['title', 'detail', 'combined_text']
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
    
    # Convert the date column to a proper datetime format, extracting only the date part
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # Group by date and calculate the average for sentiment scores
    aggregated_df = df.groupby('Date', as_index=False).mean()
    
    # Save the processed data to a new CSV file
    processed_sentiment_analysis = "processed_sentiment_analysis.csv"
    aggregated_df.to_csv(processed_sentiment_analysis, index=False)

####################Get the stock data################################
    import yfinance as yf
    # Load the sentiment analysis output
    try:
        sentiment_data = pd.read_csv(processed_sentiment_analysis)
        print("Sentiment analysis data loaded successfully.")
    except Exception as e:
        print(f"Error loading sentiment analysis data: {e}")
    
    try:
        stock_data = yf.Ticker(stock_code)
        stock_history = stock_data.history(period="1mo")  # Fetch last month's data
    
        if stock_history.empty:
            print(f"No stock data found for stock code: {stock_code}")
        else:
            # Reset index to make Date a column for merging
            stock_history.reset_index(inplace=True)
            ##########################################################################################
            # Plot the stock market price
            st.subheader(f"📊 Stock Price Chart for {stock_code}")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(stock_history["Date"], stock_history["Close"], label="Closing Price", linewidth=2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")
            ax.set_title(f"{stock_code} Stock Market Trend")
            ax.legend()
            ax.grid()

            # Show the chart in Streamlit
            st.pyplot(fig)

            # Display the latest stock data
            st.subheader("📃 Latest Stock Data")
            st.dataframe(stock_history.tail(5))  # Show the last 5 rows
            ##########################################################################################
            stock_history['Date'] = stock_history['Date'].dt.date  # Ensure Date is in the correct format
            print("Stock market data fetched successfully.")
    except Exception as e:
        print(f"Error fetching stock market data: {e}")
    
    # Merge sentiment analysis data with stock market data
    try:
        # Convert sentiment data date column to datetime if available
        if 'Date' in sentiment_data.columns:
            sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date']).dt.date
    
        # Perform an inner join on the 'Date' column
        combined_data = pd.merge(sentiment_data, stock_history, on='Date', how='inner')
        combined_data = combined_data.drop(columns=['entry_count', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])
        print("Data combined successfully.")
    
        # Save combined data to a new file
        combined_data_path = f"combined_data.csv"
        combined_data.to_csv(combined_data_path, index=False)
        print(f"Combined data saved to {combined_data_path}.")
    
    except Exception as e:
        print(f"Error combining data: {e}")

