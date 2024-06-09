import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import sklearn.metrics as sm  # Import sklearn.metrics as sm

# Title and introduction

# Background image and custom HTML titles
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
        <h1 style="color: yellow;">22AIB - INFO SQUAD</h1>
    </div>
"""
st.markdown(html_title, unsafe_allow_html=True)

html_subtitle = """
    <div style="text-align: center;">
        <h2 style="color: yellow;">EM And K-means Algorithm</h2>
    </div>
"""
st.markdown(html_subtitle, unsafe_allow_html=True)

st.write("""
This app performs clustering analysis on the Iris dataset using KMeans and Gaussian Mixture Model (GMM).
""")

# Load the dataset
dataset = load_iris()
X = pd.DataFrame(dataset.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
y = pd.DataFrame(dataset.target, columns=['Targets'])

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
colormap = np.array(['red', 'lime', 'black'])

# REAL PLOT
axes[0].scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
axes[0].set_title('Real')

# K-PLOT
model = KMeans(n_clusters=3)
model.fit(X)
predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64)
axes[1].scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY], s=40)
axes[1].set_title('KMeans')

# GMM PLOT
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns=X.columns)
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)
axes[2].scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
axes[2].set_title('GMM Classification')

st.pyplot(fig)

# Display dataset
st.subheader("Iris Dataset")
st.write(X.head())

# Display cluster results
st.subheader("Cluster Labels")
st.write("KMeans Clusters:", predY)
st.write("GMM Clusters:", y_cluster_gmm)

# Calculate and display metrics
st.subheader("Clustering Metrics")
st.write("KMeans Homogeneity Score:", sm.homogeneity_score(y.Targets, predY))
st.write("GMM Homogeneity Score:", sm.homogeneity_score(y.Targets, y_cluster_gmm))
