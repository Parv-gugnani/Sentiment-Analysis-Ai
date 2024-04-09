

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data_processing import preprocess_data
from src.model import build_model

def train_model(train_data):
    # Preprocess the data
    train_data['text'] = preprocess_data(train_data['text'])
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_data['text'], train_data['label'], test_size=0.2, random_state=42)
    
    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_val_vect = vectorizer.transform(X_val)
    
    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_vect, y_train)
    
    # Evaluate the model
    val_preds = model.predict(X_val_vect)
    val_accuracy = accuracy_score(y_val, val_preds)
    print("Validation Accuracy:", val_accuracy)
    
    return model
