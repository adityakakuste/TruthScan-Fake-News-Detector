import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Page config
st.set_page_config(
    page_title="TruthScan - Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 TruthScan")
st.subheader("AI Fake News Detection System")

st.write(
    "Enter a news headline or article below and our AI model will "
    "predict whether the news is **Fake** or **Real**."
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("fake_news.csv")
    return df

df = load_data()

# 🔥 IMPORTANT FIXES
df = df[['text', 'label']]          # keep only needed columns
df.dropna(inplace=True)             # remove empty rows
df['text'] = df['text'].astype(str)

# Normalize labels
df['label'] = df['label'].str.lower().map({'fake': 0, 'real': 1})

X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('nb', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# UI input
news_input = st.text_area("📝 Enter News Text", height=200)

if st.button("🔍 Check Authenticity"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        prediction = model.predict([news_input])[0]
        probability = model.predict_proba([news_input]).max()

        if prediction == 0:
            st.error(f"❌ Fake News\n\nConfidence: {probability*100:.2f}%")
        else:
            st.success(f"✅ Real News\n\nConfidence: {probability*100:.2f}%")

# Info
st.markdown("---")
st.markdown("### 🧠 How It Works")
st.markdown("""
- Text cleaned and converted using **TF-IDF**
- **Naive Bayes** classifier predicts authenticity
- Model trained on real & fake news dataset
""")

st.markdown("### 🛠 Tech Stack")
st.markdown("""
- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
""")

st.caption("ASEP Project | First Year Engineering")
