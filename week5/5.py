import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def main():
    bg_img = '''
        <style>
        .stApp {
            background-image: url("https://conciliac.com/wp-content/uploads/2023/07/nota-2-julio-1.webp");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        '''
    html_title = """
        <div style="text-align: center;">
            <h1 style="color: white;">22AIB - INFO SQUAD</h1>
        </div>
    """
    html_subtitle = """
        <div style="text-align: center;">
            <h2 style="color: white;">Naive Bayesian Classifier</h2>
        </div>
    """

    st.markdown(bg_img, unsafe_allow_html=True)  # Apply background image
    st.markdown(html_title, unsafe_allow_html=True)  # Display HTML title
    st.markdown(html_subtitle, unsafe_allow_html=True)  # Display HTML subtitle
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Map data CATEGORY to numeric values
        data["CATEGORY"] = data["CATEGORY"].map({"b": 1, "t": 2, "e": 3, "m": 4})

        # Replace nan values with empty string
        data = data.fillna("")

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            data["TITLE"].values, data["CATEGORY"].values, test_size=0.3, random_state=0
        )

        # Extract features from the training data using CountVectorizer
        count_vectorizer = CountVectorizer()
        count_train = count_vectorizer.fit_transform(X_train)

        # Extract features from the test data using CountVectorizer
        count_test = count_vectorizer.transform(X_test)

        # Train multinomial Naive Bayes model using CountVectorizer
        nb_classifier = MultinomialNB()
        nb_classifier.fit(count_train, y_train)

        # Predict the test set using the multinomial Naive Bayes model
        y_pred = nb_classifier.predict(count_test)

        # Print the accuracy
        st.write("Accuracy Score:", accuracy_score(y_test, y_pred))

        # Print the classification report
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))

        # Print the confusion matrix
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
