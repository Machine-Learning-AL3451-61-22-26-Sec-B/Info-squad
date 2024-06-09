import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Define the data
X = np.arange(0, 6*(np.pi), 0.1)
Y = np.sin(X)

# Scatter plot function
def scatter_plot(X, Y):
    plt.scatter(X, Y, alpha=0.2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of Y = sin(X)')

# Local Regression function
def local_regression(x0, X, Y, tau): 
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]
    xw = X.T * radial_kernel(x0, X, tau)
    beta = np.linalg.pinv(xw @ X) @ xw @ Y
    return x0 @ beta

# Radial Kernel function
def radial_kernel(x0, X, tau): 
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau ** 2)))

# Plot LWR function
def plot_lwr(tau):
    domain = np.arange(0, 6*(np.pi), 0.1)
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plt.scatter(X, Y, alpha=0.3)
    plt.plot(domain, prediction, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Local Weighted Regression with tau={tau}')
    return plt

# Streamlit app

# Background image and custom title/subtitle styles
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
html_title = """
    <div style="text-align: center;">
        <h1 style="color: yellow;">22AIB - INFO SQUAD</h1>
    </div>
"""
html_subtitle = """
    <div style="text-align: center;">
        <h2 style="color: yellow;">Local Weighted Regression Visualization</h2>
    </div>
"""
# Add HTML and CSS to the Streamlit app
st.markdown(bg_img, unsafe_allow_html=True)
st.markdown(html_title, unsafe_allow_html=True)
st.markdown(html_subtitle, unsafe_allow_html=True)
st.write("This app visualizes Local Weighted Regression on a sine function.")

# Tau input
tau = st.slider('Select tau value', 0.01, 1.0, 0.1)

# Plot the LWR
fig, ax = plt.subplots()
scatter_plot(X, Y)
plot_lwr(tau)
st.pyplot(fig)
