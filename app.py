
from flask import Flask, request, jsonify
import joblib

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return '✅ Mental Health Predictor is live!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Convert text to features and predict
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = "SuicideWatch" if pred == 1 else "Depression"

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run()
