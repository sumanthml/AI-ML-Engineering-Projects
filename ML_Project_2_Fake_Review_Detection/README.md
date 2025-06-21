# 📦 ML Project 2: Fake Review Detection using NLP & Machine Learning

This project builds a machine learning model to detect **fake product reviews** using **NLP** and **Logistic Regression**. It uses text preprocessing, TF-IDF vectorization, and classification techniques to predict whether a review is genuine or fake.

---

## 🧠 Project Goals
- Clean and process customer reviews
- Convert text into numeric features using TF-IDF
- Train a binary classifier (Logistic Regression)
- Evaluate with accuracy, classification report & confusion matrix

---

## 🗂️ Project Structure

```
ML_Project_2_Fake_Review_Detection/
├── fake_review_detection.ipynb     # Main Jupyter notebook
├── fake_reviews.csv                # Dataset (contains genuine + added fake samples)
├── accuracy.png                    # Confusion matrix heatmap
├── requirements.txt                # All required Python packages
└── README.md                       # This file
```

---

## 📊 Dataset

- Original dataset had only genuine reviews
- For demo purposes, ~100 fake reviews were added manually to make it a binary classification problem
- Text samples were balanced and shuffled before training

---

## 🔧 Tools & Libraries Used

- `pandas`, `numpy`
- `scikit-learn` (LogisticRegression, metrics, TfidfVectorizer)
- `matplotlib`, `seaborn`
- `Jupyter Notebook`

---

## 🧪 Model Evaluation

- Accuracy: *insert your actual score here*
- Precision/Recall scores included in the classification report
- Confusion matrix saved as `accuracy.png`

---

## 📌 How to Run

1. Clone this repo  
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook fake_review_detection.ipynb
   ```
4. Run all cells

---

## 💡 Future Improvements
- Use a real balanced dataset with actual fake reviews
- Try advanced models (SVM, XGBoost)
- Use word embeddings (Word2Vec, BERT)

---

## 🤝 Acknowledgements

Fake review detection is a valuable step in e-commerce trust building. This mini-project demonstrates how basic NLP + ML can be used effectively to tackle such real-world problems.

