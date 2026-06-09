import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Page Configuration ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

# --- Load Model and Vectorizer ---
@st.cache_resource # Cache the model loading for performance
def load_model():
    try:
        with open('logistic_regression_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('tfidf_vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or vectorizer files not found. Please ensure 'logistic_regression_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
        st.stop()

model, vectorizer = load_model()

# --- Text Cleaning Function (must be consistent with training) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z ]', '', text) # Keep only letters and spaces
    return text

# --- Prediction Function ---
def predict_sentiment(text):
    if not text.strip(): # Handle empty input
        return "Please enter some text for analysis."
    cleaned_text = clean_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

# --- Streamlit UI ---
st.title("Movie Review Sentiment Analyzer 😊")
st.markdown("Enter a movie review below to classify its sentiment as positive or negative.")

user_input = st.text_area("Enter your movie review here:", height=150, placeholder="Type your review...")

if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        if sentiment == 'positive':
            st.success(f"The sentiment of the review is: **{sentiment.upper()}**")
        elif sentiment == 'negative':
            st.error(f"The sentiment of the review is: **{sentiment.upper()}**")
        else:
            st.info(f"Result: {sentiment}") # For the 'Please enter some text' case
    else:
        st.warning("Please enter a review to analyze.")

st.markdown("--- Developed using Python, scikit-learn, and Streamlit --- ")
