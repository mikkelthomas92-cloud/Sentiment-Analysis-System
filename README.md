# Sentiment Analysis System

## Overview

This project is a Machine Learning-based Sentiment Analysis System developed using Python. It classifies movie reviews as **Positive** or **Negative** using the IMDb 50K Movie Reviews Dataset.

The project demonstrates the complete Machine Learning workflow, including data collection, preprocessing, feature extraction, model training, evaluation, model saving, and sentiment prediction.

---

## Features

- Binary Sentiment Classification (Positive / Negative)
- Text Preprocessing
- TF-IDF Feature Extraction
- Logistic Regression Classifier
- Trained Model Saving using Joblib
- Command-Line Interface for Prediction
- Separate Testing Script for Saved Model

---

## Dataset

**Dataset Used:**
IMDb 50K Movie Reviews Dataset

**Download Link:**
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

After downloading:

1. Extract the ZIP file.
2. Copy **IMDB Dataset.csv** into the project folder.

---

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Project Structure

```
Sentiment-Analysis-System/
│
├── sentiment_analysis.py
├── test_model.py
├── sentiment_model.pkl
├── requirements.txt
├── README.md
├── Internship project report-2.pdf
└── IMDB Dataset.csv (Download separately from Kaggle)
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mikkelthomas92-cloud/Sentiment-Analysis-System.git
```

Move into the project folder:

```bash
cd Sentiment-Analysis-System
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Train the model:

```bash
python sentiment_analysis.py
```

The program will:

- Load the dataset
- Clean the text
- Train the Logistic Regression model
- Evaluate the model
- Save the trained model as `sentiment_model.pkl`

---

## Testing the Saved Model

Run:

```bash
python test_model.py
```

Example:

```
Enter a review:

This movie was amazing.

Predicted Sentiment:
Positive
```

---

## Model Performance

**Algorithm:** Logistic Regression

**Feature Extraction:** TF-IDF Vectorizer

**Accuracy Achieved:**

**89.52%**

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Example Predictions

### Positive Review

Input

```
This movie was absolutely amazing. I loved every minute.
```

Output

```
Positive
```

### Negative Review

Input

```
Worst movie ever. Waste of time.
```

Output

```
Negative
```

---

## Internship Deliverables

This repository contains:

- Python Source Code
- Trained Machine Learning Model
- Testing Script
- Requirements File
- Project Report
- Documentation

---

## Future Improvements

- Flask Web Application
- Deep Learning Models (LSTM/BERT)
- Neutral Sentiment Detection
- Multilingual Sentiment Analysis
- Cloud Deployment

---

## Author

**Mikkel Thomas**

Computer Science and Engineering

Machine Learning Internship Project

---

## License

This project is developed for educational and internship purposes.

---

## Acknowledgement

I would like to thank **Transen Dynamics** for providing me with the opportunity to work on this internship project. It helped me gain practical experience in Machine Learning, Natural Language Processing (NLP), and Python development.
