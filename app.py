from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

def preprocess_text(text):
    """Preprocess text for fake news detection"""
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def load_or_train_model():
    """Load existing model or train a new one"""
    global model, vectorizer

    model_path = 'fake_news_model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'

    # Try to load existing model
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Model loaded successfully!")
    else:
        # Train new model if dataset exists
        if os.path.exists('crowd_sourced_balanced_dataset.csv'):
            print("Training new model...")
            df = pd.read_csv('crowd_sourced_balanced_dataset.csv')

            # Assuming the dataset has 'text' and 'label' columns
            # Adjust column names based on your actual dataset
            text_column = df.columns[0] if 'text' not in df.columns else 'text'
            label_column = df.columns[1] if 'label' not in df.columns else 'label'

            # Preprocess text
            df['processed_text'] = df[text_column].apply(preprocess_text)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'],
                df[label_column],
                test_size=0.2,
                random_state=42
            )

            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_vec, y_train)

            # Evaluate
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model trained! Accuracy: {accuracy:.4f}")

            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
        else:
            print("No dataset or model found. Please train a model first.")

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if news is fake or real"""
    try:
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500

        # Get text from request
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess and vectorize
        processed_text = preprocess_text(text)
        text_vectorized = vectorizer.transform([processed_text])

        # Predict
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]

        # Get confidence score
        confidence = max(probability) * 100

        result = {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': f'{confidence:.2f}%',
            'probabilities': {
                'real': f'{probability[0] * 100:.2f}%',
                'fake': f'{probability[1] * 100:.2f}%'
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """Retrain the model"""
    try:
        load_or_train_model()
        return jsonify({'message': 'Model trained successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load or train model on startup
    load_or_train_model()

    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
