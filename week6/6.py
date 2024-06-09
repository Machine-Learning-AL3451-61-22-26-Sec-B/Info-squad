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
        background-image: url("https://img.freepik.com/free-photo/businessman-working-futuristic-office_23-2151003702.jpg?t=st=1717914299~exp=1717917899~hmac=8dc2e270534039993e17cc88b32833e546847de3ddbab1d089da8e3fb915c83d&w=996");
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
st.write("This app uses a Bayesian Network to classify COVID-19 symptoms based on a standard WHO dataset.")

# Function to create synthetic data (simulating WHO dataset structure)
def create_synthetic_who_data():
    np.random.seed(42)
    size = 1000
    data = {
        'Fever': np.random.randint(2, size=size),
        'Cough': np.random.randint(2, size=size),
        'RunnyNose': np.random.randint(2, size=size),
        'SoreThroat': np.random.randint(2, size=size),
        'Fatigue': np.random.randint(2, size=size),
        'Diarrhoea': np.random.randint(2, size=size),
        'DifficultyBreathing': np.random.randint(2, size=size),
        'LossOfTasteOrSmell': np.random.randint(2, size=size),
        'Headache': np.random.randint(2, size=size),
        'COVID19': np.random.randint(2, size=size),
    }
    df = pd.DataFrame(data)
    return df

# Generate the synthetic dataset
df = create_synthetic_who_data()

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
    ('Fever', 'Fatigue'),
    ('Cough', 'SoreThroat'),
    ('RunnyNose', 'Headache'),
    ('Diarrhoea', 'COVID19'),
    ('DifficultyBreathing', 'Fever'),
    ('LossOfTasteOrSmell', 'COVID19'),
    ('COVID19', 'Cough'),
    ('SoreThroat', 'Headache'),
    ('Fever', 'COVID19')
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

symptoms = ['Fever', 'Cough', 'RunnyNose']
evidence = {}
for symptom in symptoms:
    value = st.selectbox(f"Do you have {symptom}?", ('Yes', 'No', 'Not Sure'), key=symptom)
    if value == 'Yes':
        evidence[symptom] = 1
    elif value == 'No':
        evidence[symptom] = 0

if evidence:
    target_result = infer.query(variables=["COVID19"], evidence=evidence)
    st.write("Probability of having COVID-19 given the symptoms:")
    st.write(target_result)
else:
    st.write("Please provide evidence for symptoms to get a prediction.")
