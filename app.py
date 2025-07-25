import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("🧠 Suicide/Depression Text Classifier")
st.write("Enter a sentence below, and the model will predict whether it's SuicideWatch or Depression.")

# Text input
user_input = st.text_area("Enter your message:")

# Predict button
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        features = vectorizer.transform([user_input])
        pred = model.predict(features)[0]
        label = "🛑 SuicideWatch" if pred == 1 else "💬 Depression"
        st.success(f"Prediction: **{label}**")
