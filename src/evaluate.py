

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_processing import preprocess_data

def evaluate_model(test_data, model):
    # Preprocess the test data
    test_data['text'] = preprocess_data(test_data['text'])
    
    # Vectorize text data using TF-IDF
    X_test_vect = vectorizer.transform(test_data['text'])
    
    # Evaluate the trained model on test data
    test_preds = model.predict(X_test_vect)
    test_accuracy = accuracy_score(test_data['label'], test_preds)
    print("Test Accuracy:", test_accuracy)
    
    return test_accuracy
