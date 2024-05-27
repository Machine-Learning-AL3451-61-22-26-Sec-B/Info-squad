import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.tree import export_graphviz 
import graphviz 
import streamlit as st 
import missingno as mn
import io

# Set the page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Sidebar for file upload and model selection
st.sidebar.title("Upload and Configure")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Header for the main page
st.title("Credit Card Fraud Detection")

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)

    # Show the first five rows of the dataset
    st.header("Dataset Preview")
    st.write(df.head())

    # Display dataset information
    st.header("Dataset Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Find and display missing values
    st.header("Missing Values")
    st.write(df.isna().sum())

    # Visualize missing values
    st.subheader("Missing Values Visualization")
    mn.bar(df)
    st.pyplot()

    X = df.drop('fraud', axis=1) 
    Y = df['fraud'] 

    st.header("Feature Analysis")
    st.subheader("Boxplot of Features")
    plt.figure(figsize=(20,20)) 
    plotnumber = 1
    for col in X.columns: 
        if plotnumber <= 7: 
            ax = plt.subplot(5,5,plotnumber) 
            sns.boxplot(X[col]) 
            plt.xlabel(col, fontsize=20) 
            plotnumber += 1
    plt.tight_layout() 
    st.pyplot()

    # Split X and Y 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Standardize the data 
    scaler = StandardScaler() 
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.transform(X_test) 

    # Decision Tree classifier 
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0) 
    clf.fit(X_train, Y_train) 

    # Predict the values of testing 
    Y_pred_dt = clf.predict(X_test) 

    # Calculate and display the confusion matrix
    st.header("Model Performance")
    st.subheader("Decision Tree")
    st.write("Confusion Matrix:")
    cm_dt = confusion_matrix(Y_test, Y_pred_dt) 
    st.write(cm_dt)

    # Calculate and display the accuracy score
    accuracy_dt = accuracy_score(Y_test, Y_pred_dt) 
    st.write("Accuracy:", accuracy_dt)

    # Visualize the decision tree
    st.write("Decision Tree Visualization:")
    dot_data = export_graphviz(clf, feature_names=X.columns, filled=True)
    st.graphviz_chart(dot_data)

    # Naive Bayes Algorithm 
    clf1 = GaussianNB() 

    # Train the model 
    clf1.fit(X_train, Y_train) 

    # Predict the values of testing 
    Y_pred_nb = clf1.predict(X_test) 

    # Calculate and display the confusion matrix
    st.subheader("Naive Bayes")
    st.write("Confusion Matrix:")
    cm_nb = confusion_matrix(Y_test, Y_pred_nb) 
    st.write(cm_nb)

    # Calculate and display the accuracy score
    accuracy_nb = accuracy_score(Y_test, Y_pred_nb) 
    st.write("Accuracy:", accuracy_nb)

    # Comparing the accuracy scores 
    accuracies = [accuracy_dt * 100, accuracy_nb * 100] 
    labels = ['Decision Tree', 'Naive Bayes'] 

    # Display the accuracy comparison bar chart
    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots()
    ax.bar(labels, accuracies, color=['blue', 'green']) 
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Comparison of Model Accuracies')
    st.pyplot(fig)
else:
    st.info("Please upload a CSV file to proceed.")
