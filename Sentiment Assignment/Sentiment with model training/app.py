# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from main import clean_text, train_model

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="Amazon Reviews Sentiment Dashboard", layout="wide")

st.title("📦 Amazon Reviews Sentiment Dashboard")

# ------------------------------
# Load model
# ------------------------------
model, tfidf, le, df_train = train_model("Kaggle dataset.csv")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV with 'body' column", type=["csv"])
show_wordclouds = st.sidebar.checkbox("Show WordClouds", value=True)
show_distribution = st.sidebar.checkbox("Show Sentiment Distribution", value=True)

# ------------------------------
# Load data (default or uploaded)
# ------------------------------
if uploaded_file:
    amazon_df = pd.read_csv(uploaded_file)
else:
    amazon_df = pd.read_csv("raw_reviews.csv")

amazon_df = amazon_df.dropna(subset=['body'])
amazon_df['processed_review'] = amazon_df['body'].apply(clean_text)

# ------------------------------
# Predict sentiment
# ------------------------------
X = tfidf.transform(amazon_df['processed_review'])
predictions = model.predict(X)
amazon_df['predicted_sentiment'] = le.inverse_transform(predictions)

sentiment_counts = amazon_df['predicted_sentiment'].value_counts()

# ------------------------------
# SENTIMENT COUNT CARDS
# ------------------------------
st.subheader("🔢 Sentiment Summary")

col1, col2, col3 = st.columns(3)

col1.metric("😊 Positive", sentiment_counts.get("positive", 0))
col2.metric("😐 Neutral", sentiment_counts.get("neutral", 0))
col3.metric("😠 Negative", sentiment_counts.get("negative", 0))

# ------------------------------
# Charts Section
# ------------------------------
if show_distribution:
    st.subheader("📊 Sentiment Distribution")

    col4, col5 = st.columns(2)

    with col4:
        st.write("**Bar Chart**")
        st.bar_chart(sentiment_counts)

    with col5:
        st.write("**Pie Chart**")
        fig, ax = plt.subplots()
        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

# ------------------------------
# WordCloud Section
# ------------------------------
if show_wordclouds:
    st.subheader("☁️ WordCloud Analysis")

    # All Reviews
    st.write("### All Reviews")
    all_text = " ".join(amazon_df['body'])
    wc_all = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.imshow(wc_all, interpolation="bilinear")
    ax1.axis("off")
    st.pyplot(fig1)

    col6, col7, col8 = st.columns(3)

    # Positive
    with col6:
        st.write("### 😊 Positive")
        pos_text = " ".join(amazon_df[amazon_df['predicted_sentiment'] == "positive"]['body'])
        wc_pos = WordCloud(width=400, height=300, background_color="white").generate(pos_text)
        fig2, ax2 = plt.subplots()
        ax2.imshow(wc_pos)
        ax2.axis("off")
        st.pyplot(fig2)

    # Neutral
    with col7:
        st.write("### 😐 Neutral")
        neu_text = " ".join(amazon_df[amazon_df['predicted_sentiment'] == "neutral"]['body'])
        wc_neu = WordCloud(width=400, height=300, background_color="white").generate(neu_text)
        fig3, ax3 = plt.subplots()
        ax3.imshow(wc_neu)
        ax3.axis("off")
        st.pyplot(fig3)

    # Negative
    with col8:
        st.write("### 😠 Negative")
        neg_text = " ".join(amazon_df[amazon_df['predicted_sentiment'] == "negative"]['body'])
        wc_neg = WordCloud(width=400, height=300, background_color="white").generate(neg_text)
        fig4, ax4 = plt.subplots()
        ax4.imshow(wc_neg)
        ax4.axis("off")
        st.pyplot(fig4)

# ------------------------------
# Table preview
# ------------------------------
st.subheader("📄 Predicted Reviews (Preview)")
st.dataframe(amazon_df[['body', 'predicted_sentiment']].head(15))

# ------------------------------
# Single review prediction
# ------------------------------
st.subheader("✍️ Predict Single Review")

review = st.text_area("Enter your review:")
if st.button("Predict"):
    if review.strip():
        processed = clean_text(review)
        pred = model.predict(tfidf.transform([processed]))
        sentiment = le.inverse_transform(pred)[0]
        st.success(f"Predicted Sentiment: **{sentiment.upper()}**")
    else:
        st.warning("Please enter a review.")
