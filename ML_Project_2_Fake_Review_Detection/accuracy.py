import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import string



# Load the dataset
df = pd.read_csv("fake_reviews_dataset.csv")  # Or whatever your filename is
df.head()



print(df.columns)        # Check column names
print(df['label'].value_counts())  # Assuming 'label' column contains "FAKE"/"GENUINE"




print(df.columns)



df['cleaned_review'] = df['text'].apply(clean_text)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)



X = tfidf.fit_transform(df['cleaned_review'])
y = df['label'].apply(lambda x: 1 if x == "FAKE" else 0)


from sklearn.model_selection import train_test_split

# Split 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)





print("Full dataset:\n", y.value_counts())
print("\nTrain set:\n", y_train.value_counts())
print("\nTest set:\n", y_test.value_counts())
#output for the abouve three line code
Full dataset:
 label
0    40526
Name: count, dtype: int64

Train set:
 label
0    32420
Name: count, dtype: int64

Test set:
 label
0    8106
Name: count, dtype: int64



# Add 50 fake reviews manually
fake_reviews = pd.DataFrame({
    'text': ["This product is amazing! Highly recommended."] * 50,
    'label': [1] * 50
})

# Append to your current dataset
df = pd.concat([df, fake_reviews], ignore_index=True)




df['cleaned_review'] = df['text'].apply(clean_text)




# Add 100 fake reviews manually for demo/testing
fake_samples = pd.DataFrame({
    'text': [
        "Absolutely terrible product. Waste of money!",
        "This is fake. The product broke in a day.",
        "Do not buy! It's a scam item!",
        "Worst product I've ever seen. Totally fake!"
    ] * 25,  # repeat to make 100 rows
    'label': [1] * 100
})

# Combine with the original dataset
df = pd.concat([df, fake_samples], ignore_index=True)

# Shuffle the rows so fake & genuine are mixed
df = df.sample(frac=1, random_state=42).reset_index(drop=True)





df['cleaned_review'] = df['text'].apply(clean_text)

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(df['cleaned_review'])
y = df['label']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# evaluation
Accuracy: 0.8688544739429696

Classification Report:
               precision    recall  f1-score   support

           0       0.85      0.89      0.87      4047
           1       0.89      0.85      0.87      4089

    accuracy                           0.87      8136
   macro avg       0.87      0.87      0.87      8136
weighted avg       0.87      0.87      0.87      8136




import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap='coolwarm')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("accuracy.png") 
plt.show()







