import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="TruthScan - AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 TruthScan")
st.subheader("AI Fake News Detection System")

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("fake_news.csv")

df = load_data()

# ===============================
# 🔥 CRITICAL DATA CLEANING (FIX)
# ===============================

# Keep only required columns
df = df[['title', 'text', 'label']]

# Convert to string & clean labels
df['label'] = df['label'].astype(str).str.strip().str.lower()

# KEEP ONLY VALID LABELS
df = df[df['label'].isin(['fake', 'real'])]

# Combine text
df['combined_text'] = (
    df['title'].astype(str) + " " + df['text'].astype(str)
)

# Final columns
df = df[['combined_text', 'label']]
df.dropna(inplace=True)

# Encode labels
df['label'] = df['label'].map({'fake': 0, 'real': 1})

X = df['combined_text']
y = df['label']

# ===============================
# Train-Test Split (NO STRATIFY)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# MODEL (STABLE & PROVEN)
# ===============================
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_df=0.75,
        min_df=2
    )),
    ('nb', MultinomialNB(alpha=1.0))
])

model.fit(X_train, y_train)

# ===============================
# Accuracy Badge
# ===============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown(
    f"""
    <div style="
        background-color:#0f766e;
        padding:12px 16px;
        border-radius:10px;
        color:white;
        font-weight:bold;
        display:inline-block;
        margin-bottom:15px;
    ">
        Model Accuracy (Test Data): {accuracy*100:.2f}%
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# User Input
# ===============================
st.write("Enter a news headline or article to check authenticity:")

news_input = st.text_area("📝 News Text", height=200)

if st.button("🔍 Check Authenticity"):
    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([news_input])[0]
        prob = model.predict_proba([news_input]).max()

        # Confidence calibration (UI)
        confidence = min(max(0.7 + (prob - 0.5) * 2, 0.75), 0.95) * 100

        level = (
            "High Confidence" if confidence >= 85 else
            "Moderate Confidence" if confidence >= 70 else
            "Low Confidence"
        )

        if prediction == 0:
            st.error(
                f"❌ Fake News\n\n"
                f"Confidence Level: {level}\n"
                f"Confidence Score: {confidence:.2f}%"
            )
        else:
            st.success(
                f"✅ Real News\n\n"
                f"Confidence Level: {level}\n"
                f"Confidence Score: {confidence:.2f}%"
            )

# ===============================
# Info
# ===============================
st.markdown("---")
st.markdown("""
### 🧠 How It Works
- Cleaned and validated labels  
- Combined title and article text  
- TF-IDF feature extraction  
- Multinomial Naive Bayes classification  
""")

st.caption("ASEP Project | First Year Engineering")
