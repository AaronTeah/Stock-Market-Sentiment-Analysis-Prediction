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
        latest_news = df.sort_values(by='Date', ascending=False).head(5)
    
        # Display the latest 5 unique news articles with sentiment scores
        for i, row in latest_news.iterrows():
            st.markdown(f"### {i+1}. {row['title']}")
            st.write(f"📅 **Date:** {row['Date'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"📰 **Summary:** {row['detail']}")
            
            # Show sentiment scores
            st.write(f"👍 **Positive:** `{row['positive']:.2f}` | 😐 **Neutral:** `{row['neutral']:.2f}` | 👎 **Negative:** `{row['negative']:.2f}`")
            st.write(f"🧮 **Overall Sentiment Score:** {row['score']:.5f}")
            st.write("---")  # Divider for clarity
    
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")

        
#st.write(f"👍 **Positive:** `{row['positive']:.2f}` | 😐 **Neutral:** `{row['neutral']:.2f}` | 👎 **Negative:** `{row['negative']:.2f}`")
