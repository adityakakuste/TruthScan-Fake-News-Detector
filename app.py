import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="TruthScan – AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 TruthScan")
st.subheader("AI Fake News Detection System")

st.write(
    "Enter a news headline or article to check whether it is **Fake** or **Real** "
    "using an AI-based text classification model."
)

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("fake_news.csv")

# ===============================
# Data Cleaning & Preparation
# ===============================
df = df[['title', 'text', 'label']]

df['label'] = df['label'].astype(str).str.strip().str.lower()
df = df[df['label'].isin(['fake', 'real'])]

df['combined_text'] = df['title'].astype(str) + " " + df['text'].astype(str)
df = df[['combined_text', 'label']]
df.dropna(inplace=True)

df['label'] = df['label'].map({'fake': 0, 'real': 1})

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df['combined_text']
y = df['label']

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# ML Model Pipeline
# ===============================
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=5,
        sublinear_tf=True
    )),
    ('nb', MultinomialNB(alpha=0.7))
])

model.fit(X_train, y_train)

# ===============================
# User Input
# ===============================
news_input = st.text_area(
    "📝 Enter News Text",
    height=200,
    placeholder="Paste a news headline or article here..."
)

if st.button("🔍 Check Authenticity"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        prediction = model.predict([news_input])[0]

        if prediction == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")

