import streamlit as st

# Streamlit App Title
st.title("Stock Sentiment Analysis & Prediction")

# Create a form for user input
with st.form(key="stock_form"):
    st.subheader("Enter Stock Information")
    
    # User inputs for stock name and code
    stock_name = st.text_input("Stock Name (e.g., KLCI, Apple, Tesla)", value="KLCI")
    stock_code = st.text_input("Stock Code (e.g., ^KLSE, AAPL, TSLA)", value="^KLSE")
    
    # Submit button
    submit_button = st.form_submit_button(label="Submit")

# Process user input after submission
if submit_button:
    st.success(f"You have selected: **{stock_name}** with stock code **{stock_code}**")
    
    # You can now pass `stock_name` and `stock_code` into your scraping and prediction functions.
    st.write("Proceeding with data scraping and sentiment analysis...")
    
    # Call your functions here, for example:
    # scrape_stock_data(stock_name, stock_code)
