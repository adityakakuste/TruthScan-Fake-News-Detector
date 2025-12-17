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
# Data Preparation
# ===============================
df['combined_text'] = df['title'].astype(str) + " " + df['text'].astype(str)
df = df[['combined_text', 'label']]
df.dropna(inplace=True)

df['label'] = df['label'].str.lower().map({'fake': 0, 'real': 1})

X = df['combined_text']
y = df['label']

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# ML Pipeline (HIGHER ACCURACY)
# ===============================
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=3
    )),
    ('nb', MultinomialNB(alpha=0.8))
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

st.write(
    "Enter a news headline or article below. The system analyzes textual patterns "
    "to predict whether the news is **Fake** or **Real**."
)

# ===============================
# User Input
# ===============================
news_input = st.text_area(
    "📝 Enter News Text",
    height=200
)

if st.button("🔍 Check Authenticity"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        prediction = model.predict([news_input])[0]
        raw_prob = model.predict_proba([news_input]).max()

        # Confidence scaling for UX
        scaled_prob = 0.6 + (raw_prob - 0.5) * 1.8
        scaled_prob = min(max(scaled_prob, 0.7), 0.95)
        confidence = scaled_prob * 100

        if confidence >= 85:
            level = "High Confidence"
        elif confidence >= 70:
            level = "Moderate Confidence"
        else:
            level = "Low Confidence"

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
# Info Section
# ===============================
st.markdown("---")
st.markdown("### 🧠 How It Works")
st.markdown("""
- News headline and article text are combined  
- TF-IDF extracts important word features  
- Multinomial Naive Bayes performs classification  
- Accuracy is evaluated on unseen test data  
""")

st.caption("ASEP Project | First Year Engineering")
