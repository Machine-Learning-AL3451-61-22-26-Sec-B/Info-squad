import pandas as pd
import numpy as np
from collections import Counter
import streamlit as st

# Calculate entropy
def entropy(y):
    class_counts = Counter(y)
    total_instances = len(y)
    entropy_value = -sum((count/total_instances) * np.log2(count/total_instances) for count in class_counts.values())
    return entropy_value

# Calculate information gain
def information_gain(X, y, attribute):
    total_entropy = entropy(y)
    values, counts = np.unique(X[attribute], return_counts=True)
    
    weighted_entropy = sum((counts[i]/sum(counts)) * entropy(y[X[attribute] == values[i]]) for i in range(len(values)))
    gain = total_entropy - weighted_entropy
    return gain

# ID3 algorithm
def id3(X, y, attributes):
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    
    if len(attributes) == 0:
        return Counter(y).most_common()[0][0]
    
    gains = [information_gain(X, y, attribute) for attribute in attributes]
    best_attr = attributes[np.argmax(gains)]
    
    tree = {best_attr: {}}
    for value in np.unique(X[best_attr]):
        sub_X = X[X[best_attr] == value].drop(columns=[best_attr])
        sub_y = y[X[best_attr] == value]
        subtree = id3(sub_X, sub_y, [attr for attr in attributes if attr != best_attr])
        tree[best_attr][value] = subtree
    
    return tree

# Classify a new sample
def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    if sample[attribute] in tree[attribute]:
        return classify(tree[attribute][sample[attribute]], sample)
    else:
        return None

# Streamlit app
def main():
    # Inject CSS for background image
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://cdn-gcp.new.marutitech.com/robot_humanoid_using_tablet_computer_big_data_analytic_1_94eab7101e.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    '''
    # Inject CSS with st.markdown
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    # Display the titles using st.markdown
    html_title = """
        <div style="text-align: center;">
            <h1 style="color: red;">22AIB - INFO SQUAD</h1>
        </div>
    """
    html_subtitle = """
        <div style="text-align: center;">
            <h2 style="color: red;">ID3 Decision Tree Classifier</h2>
        </div>
    """
    st.markdown(html_title, unsafe_allow_html=True)
    st.markdown(html_subtitle, unsafe_allow_html=True)


    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df)

        # Specify the target column
        target_column = st.selectbox("Select the target column", df.columns)
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            attributes = X.columns.tolist()
            tree = id3(X, y, attributes)
            
            st.write("Decision Tree:")
            st.write(tree)
            
            st.write("Classify a new sample:")
            sample = {}
            for attr in attributes:
                sample[attr] = st.selectbox(f"Select {attr}", df[attr].unique())
                
            if st.button("Classify"):
                result = classify(tree, sample)
                st.write(f"The class is: {result}")

if __name__ == "__main__":
    main()
