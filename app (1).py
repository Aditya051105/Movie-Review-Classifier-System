import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Define the clean_text function (must be the same as used during training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z ]', '', text) # Keep only letters and spaces
    return text

# Function to predict sentiment
def predict_sentiment(text):
    # Clean the input text
    cleaned_text = clean_text(text)
    # Transform the text using the loaded TF-IDF vectorizer
    text_vectorized = vectorizer.transform([cleaned_text])
    # Predict the sentiment using the loaded model
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Example usage (for demonstration if run as a script)
if __name__ == '__main__':
    print("Sentiment Prediction Application")
    print("------------------------------")

    while True:
        user_input = input("Enter a review (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        sentiment = predict_sentiment(user_input)
        print(f"Predicted sentiment: {sentiment}\n")

    print("Application stopped.")
