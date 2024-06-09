import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class CandidateElimination(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.specific_h = None
        self.general_h = None

    def fit(self, X, y):
        concepts = np.array(X)
        target = np.array(y)

        self.specific_h = concepts[0].copy()
        self.general_h = [["?" for _ in range(len(self.specific_h))] for _ in range(len(self.specific_h))]

        for i, h in enumerate(concepts):
            if target[i] == 1:
                for x in range(len(self.specific_h)):
                    if h[x] != self.specific_h[x]:
                        self.specific_h[x] = "?"
                        self.general_h[x][x] = "?"
            else:
                for x in range(len(self.specific_h)):
                    if h[x] != self.specific_h[x]:
                        self.general_h[x][x] = self.specific_h[x]
                    else:
                        self.general_h[x][x] = "?"

        indices = [i for i, h in enumerate(self.general_h) if h == ["?" for _ in range(len(self.specific_h))]]
        self.general_h = np.delete(self.general_h, indices, axis=0)

        return self

    def predict(self, X):
        predictions = []
        for instance in X:
            match_specific = all(
                s == "?" or s == val for s, val in zip(self.specific_h, instance)
            )
            match_general = any(
                all(
                    g == "?" or g == val for g, val in zip(hypothesis, instance)
                )
                for hypothesis in self.general_h
            )
            if match_specific and match_general:
                predictions.append(1)
            else:
                predictions.append(0)
        return np.array(predictions)

# Streamlit application

# HTML and CSS for center-aligned titles
html_title = """
    <div style="text-align: center;">
        <h1 style="color: white;">22AIB - INFO SQUAD</h1>
    </div>
"""

html_subtitle = """
    <div style="text-align: center;">
        <h2 style="color: white;">Candidate Elimination Algorithm</h2>
    </div>
"""

# CSS for background image
page_bg_img = '''
<style>
.stApp {
    background: grey;
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''

# Inject CSS with st.markdown
st.markdown(page_bg_img, unsafe_allow_html=True)
# Display the titles using st.markdown
st.markdown(html_title, unsafe_allow_html=True)
st.markdown(html_subtitle, unsafe_allow_html=True)

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Split data into features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Encode target values as binary (0 and 1)
    y = y.apply(lambda x: 1 if x == "Yes" else 0)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = CandidateElimination()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test.values)

    # Convert predictions back to original labels
    predictions_labels = np.array(["Yes" if pred == 1 else "No" for pred in predictions])

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Display results in horizontal format
    results = {
        "Predictions": predictions_labels,
        "Final Specific Hypothesis (S)": [model.specific_h],
        "Final General Hypothesis (G)": [model.general_h],
        "Accuracy": [accuracy]
    }

    results_df = pd.DataFrame(results).T
    results_df.columns = ["Value"]

    st.table(results_df)
