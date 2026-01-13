import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.read_csv("Kaggle dataset.csv")
df = df.dropna(subset=['cleaned_review'])
stop_words = set(stopwords.words('english'))

# Negation words must be kept
negations = {'not', 'no', 'nor', 'never'}
stop_words = stop_words - negations
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)       # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)    # remove punctuation & numbers
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)
df['processed_review'] = df['cleaned_review'].apply(clean_text)
df[['cleaned_review', 'processed_review']].head()
X = df['processed_review']
y = df['sentiments']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

model = OneVsRestClassifier(
    LogisticRegression()
)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
y_pred
def vectorize_text(text, tfidf):
    """
    Converts input text into TF-IDF vector using a trained TF-IDF vectorizer.

    Parameters:
    text  : str or list of str
    tfidf : trained TfidfVectorizer

    Returns:
    TF-IDF vector
    """

    # If single string is passed, convert to list
    if isinstance(text, str):
        text = [text]

    # Transform using trained TF-IDF
    vector = tfidf.transform(text)

    return vector
def predict_sentiment(review):
    review_cleaned = clean_text(review)
    review_vector = vectorize_text(review_cleaned, tfidf)
    result = model.predict(review_vector)
    return result[0]
