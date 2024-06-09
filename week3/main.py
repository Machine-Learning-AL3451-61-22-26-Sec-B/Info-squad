import numpy as np
import streamlit as st

# HTML code for setting background image and displaying titles
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
        <h1 style="color: red;">22AIB - INFO SQUAD</h1>
    </div>
"""
html_subtitle = """
    <div style="text-align: center;">
        <h2 style="color: red;">Backpropagation Algorithm</h2>
    </div>
"""

# Neural Network class
class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o 

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)
    
    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

# Streamlit app
def main():
    # Set background image and display titles
    st.markdown(bg_img, unsafe_allow_html=True)
    st.markdown(html_title, unsafe_allow_html=True)
    st.markdown(html_subtitle, unsafe_allow_html=True)

    # Get user input for sample data
    st.subheader("Enter Sample Data:")
    input_1 = st.number_input("Hours of sleep:", value=2)
    input_2 = st.number_input("Hours of study:", value=9)

    # Sample data
    X = np.array([[input_1, input_2]])
    y = np.array([[st.number_input("Test score:", value=92)]])

    # Normalize data
    X = X / np.amax(X, axis=0)
    y = y / 100

    # Instantiate the neural network
    NN = Neural_Network()

    # Display original data
    st.subheader("Original Data:")
    st.write("Input: ", X)
    st.write("Actual Output: ", y)

    # Train the neural network
    NN.train(X, y)

    # Display predicted output and loss
    st.subheader("Predicted Output after Training:")
    predicted_output = NN.forward(X)
    st.write("Predicted Output: ", predicted_output)
    loss = np.mean(np.square(y - predicted_output))
    st.write("Loss: ", loss)

if __name__ == "__main__":
    main()
