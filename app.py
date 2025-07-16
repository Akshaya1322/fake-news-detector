import streamlit as st
import pandas as pd
import string
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocessing setup
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Load and prepare data
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

true_df['label'] = 1
fake_df['label'] = 0

data = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)
data['content'] = data['title'] + " " + data['text']
data['cleaned_content'] = data['content'].apply(clean_text)

X = data['cleaned_content']
y = data['label']

vectorizer = TfidfVectorizer(max_df=0.7)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# Streamlit App
st.title("📰 AI Fake News Detector (Improved)")
st.subheader("Paste a news article to check if it's Real or Fake")

user_input = st.text_area("Enter news article (headline + paragraph):")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a news article first.")
    else:
        cleaned = clean_text(user_input)
        input_vec = vectorizer.transform([cleaned])
        prediction = model.predict(input_vec)[0]
        if prediction == 1:
            st.success("✅ This news article is likely REAL.")
        else:
            st.error("🚫 This news article is likely FAKE.")
