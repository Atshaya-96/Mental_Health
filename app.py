import streamlit as st
import joblib
from textblob import TextBlob

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Mental Health Text Classifier", layout="centered")

st.title("🧠 Mental Health Text Classifier")
st.write("This app predicts if a message is related to **Depression**, **SuicideWatch**, or just **Neutral**.")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Check sentiment
        sentiment = TextBlob(user_input).sentiment.polarity
        if sentiment > 0.2:
            st.success("Prediction: **Neutral** (positive message detected)")
        else:
            # Model prediction
            X = vectorizer.transform([user_input])
            pred = model.predict(X)[0]
            label = "SuicideWatch" if pred == 1 else "Depression"
            st.error(f"Prediction: **{label}**")
