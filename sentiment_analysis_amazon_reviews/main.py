import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK assets (run once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# 1️⃣ Load raw dataset
df = pd.read_csv("amazon_reviews.csv")
print("Raw data sample:")
print(df[['reviewText', 'overall']].head())

# 2️⃣ Keep only necessary columns and drop NaNs
df = df[['reviewText', 'overall']].dropna()

# 3️⃣ Assign sentiment labels
def label_sentiment(r):
    if r >= 4:
        return 'positive'
    elif r <= 2:
        return 'negative'
    else:
        return 'neutral'
df['sentiment'] = df['overall'].apply(label_sentiment)

# 4️⃣ Text cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|<.*?>", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t.isalpha()]
    return " ".join(tokens)

df['cleaned'] = df['reviewText'].apply(clean_text)

# 5️⃣ Split data
X = df['cleaned']; y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Feature extraction
tfidf = TfidfVectorizer(max_features=10000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 7️⃣ Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# 8️⃣ Evaluate performance
y_pred = model.predict(X_test_vec)
print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("🔹 Report:\n", classification_report(y_test, y_pred))

# 9️⃣ Confusion matrix plot
cm = confusion_matrix(y_test, y_pred, labels=['positive','neutral','negative'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['pos','neu','neg'], yticklabels=['pos','neu','neg'])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
