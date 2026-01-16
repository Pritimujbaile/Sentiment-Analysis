# main.py
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Download NLTK resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Stopwords setup
stop_words = set(stopwords.words('english'))
negations = {'not', 'no', 'nor', 'never'}
stop_words = stop_words - negations

lemmatizer = WordNetLemmatizer()

# Text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)           # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)        # remove punctuation & numbers
    text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ----------------------
# Train model function
# ----------------------
def train_model(csv_path="Kaggle dataset.csv"):
    # If model exists, load it
    if os.path.exists("logistic_model.pkl") and os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("label_encoder.pkl"):
        model = joblib.load("logistic_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        le = joblib.load("label_encoder.pkl")
        df = pd.read_csv(csv_path).dropna(subset=['body'])
        df['processed_review'] = df['body'].apply(clean_text)
        return model, tfidf, le, df

    # Otherwise, train model
    df = pd.read_csv(csv_path).dropna(subset=['body'])
    df['processed_review'] = df['body'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1,2))
    X = tfidf.fit_transform(df['processed_review'])

    le = LabelEncoder()
    y = le.fit_transform(df['sentiments'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model for future use
    joblib.dump(model, "logistic_model.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump(le, "label_encoder.pkl")

    return model, tfidf, le, df
