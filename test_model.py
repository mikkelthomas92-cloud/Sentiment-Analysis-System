import joblib
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the saved model
model = joblib.load("sentiment_model.pkl")

# Stop words
stop_words = ENGLISH_STOP_WORDS

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

print("=" * 40)
print("     SENTIMENT ANALYSIS TEST")
print("=" * 40)

while True:
    review = input("\nEnter a review (type 'exit' to quit): ")

    if review.lower() == "exit":
        print("Goodbye!")
        break

    clean_review = preprocess(review)

    prediction = model.predict([clean_review])[0]

    print("\nPredicted Sentiment:", prediction)