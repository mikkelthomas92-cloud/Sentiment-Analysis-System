import pandas as pd
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")

data = pd.read_csv("IMDB Dataset.csv/IMDB Dataset.csv")

print("Dataset Loaded Successfully!")
print(data.head())
print("Dataset Shape:", data.shape)

# -----------------------------
# Text Preprocessing
# -----------------------------
stop_words = ENGLISH_STOP_WORDS

def preprocess(text):
    text = str(text).lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Split into words
    words = text.split()

    # Remove stop words
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

print("\nCleaning text...")

data["clean_review"] = data["review"].apply(preprocess)

# -----------------------------
# Features and Labels
# -----------------------------
X = data["clean_review"]
y = data["sentiment"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model
# -----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Training Completed!")

# -----------------------------
# Prediction
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "sentiment_model.pkl")

print("\nModel Saved Successfully!")

# -----------------------------
# User Testing
# -----------------------------
print("\n==============================")
print(" SENTIMENT ANALYSIS SYSTEM ")
print("==============================")

while True:
    review = input("\nEnter Review (type 'exit' to quit): ")

    if review.lower() == "exit":
        print("Thank You!")
        break

    clean_review = preprocess(review)

    result = model.predict([clean_review])[0]

    print("Predicted Sentiment:", result)
