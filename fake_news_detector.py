import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load Data
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

true_df['label'] = 1
fake_df['label'] = 0

# 2. Combine and shuffle
data = pd.concat([true_df, fake_df])
data = data.sample(frac=1).reset_index(drop=True)

# 3. Combine title + text into one field
data['content'] = data['title'] + " " + data['text']

# 4. Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove links
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with digits
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

data['cleaned_content'] = data['content'].apply(clean_text)

# 5. Split data
X = data['cleaned_content']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Vectorize
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 7. Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 8. Predict & Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
