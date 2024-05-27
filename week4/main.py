import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Load the data from CSV file
def load_data(file):
    data = pd.read_csv("CEA.csv")
    return data

# Preprocess the data
def preprocess_data(data):
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1] # Features
    y = data.iloc[:, -1]  # Target variable

    # One-hot encode categorical variables
    X = pd.get_dummies(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Train the Naive Bayes classifier
def train_classifier(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier

# Test the classifier and compute accuracy
def test_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Streamlit app
def main():
    st.title("Naive Bayes Classifier")
   
    # File uploader
    uploaded_file = st.file_uploader("Upload your training data CSV file", type=["csv"])
   
    if uploaded_file is not None:
        # Load data
        data = load_data(uploaded_file)
       
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(data)

        # Train classifier
        classifier = train_classifier(X_train, y_train)

        # Test classifier
        accuracy = test_classifier(classifier, X_test, y_test)

        # Display accuracy
        st.write("Accuracy:", accuracy)

if _name_ == "_main_":
    main()
