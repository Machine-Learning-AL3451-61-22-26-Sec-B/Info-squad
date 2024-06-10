import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Add background image and custom HTML titles
bg_img = '''
    <style>
    .stApp {
        background: black;
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
'''
st.markdown(bg_img, unsafe_allow_html=True)

html_title = """
    <div style="text-align: center;">
        <h1 style="color: red;">22AIB - INFO SQUAD</h1>
    </div>
"""
st.markdown(html_title, unsafe_allow_html=True)

html_subtitle = """
    <div style="text-align: center;">
        <h2 style="color: red;">Bayesian Network for COVID-19 Symptom Classification</h2>
    </div>
"""
st.markdown(html_subtitle, unsafe_allow_html=True)

# Title and introduction
st.write("This app uses a Bayesian Network to classify COVID-19 symptoms based on a dataset.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the first five rows of the dataset
    st.subheader("First Five Rows of the Dataset")
    st.write(df.head())

    # Dataset description
    st.subheader("Dataset Description")
    st.write(df.describe())

    # Check for missing values
    st.subheader("Missing Values")
    st.write(df.isna().sum())

    # Display histograms for all numeric columns
    st.subheader("Histograms")
    for column in df.select_dtypes(include=np.number).columns:
        st.write(f"Histogram for {column}")
        fig, ax = plt.subplots()
        ax.hist(df[column].dropna(), bins=20)
        st.pyplot(fig)

    # Define the Bayesian Network structure
    st.subheader("Bayesian Network Structure")
    model = BayesianModel([
        ('Fever', 'Tiredness'),
        ('Dry-Cough', 'Sore-Throat'),
        ('Runny-Nose', 'Pains'),
        ('Diarrhea', 'None_Sympton'),
        ('Difficulty-in-Breathing', 'Fever'),
        ('Nasal-Congestion', 'Runny-Nose'),
        ('Sore-Throat', 'Tiredness'),
        ('Fever', 'Diarrhea')
    ])
    model.fit(df, estimator=MaximumLikelihoodEstimator)

    # Display CPD values
    st.subheader("Conditional Probability Distributions (CPDs)")
    for node in model.nodes():
        st.write(f"CPD of {node}")
        st.text(model.get_cpds(node))

    # Perform Variable Elimination for inference
    st.subheader("Inference using Variable Elimination")
    infer = VariableElimination(model)

    symptoms = ['Fever', 'Dry-Cough', 'Runny-Nose']
    evidence = {}
    for symptom in symptoms:
        value = st.selectbox(f"Do you have {symptom}?", ('Yes', 'No', 'Not Sure'), key=symptom)
        if value == 'Yes':
            evidence[symptom] = 1
        elif value == 'No':
            evidence[symptom] = 0

    if evidence:
        target_result = infer.query(variables=["None_Sympton"], evidence=evidence)
        st.write("Probability of having COVID-19 given the symptoms:")
        st.write(target_result)
    else:
        st.write("Please provide evidence for symptoms to get a prediction.")
else:
    st.write("Please upload a CSV file to proceed.")
