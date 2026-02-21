import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt



st.set_page_config(
    page_title="ID3 Decision Tree Classifier",
    layout="wide"
)



st.title("ID3 Decision Tree Classifier – Practical 2")

st.write(
    "This app demonstrates the ID3 algorithm using Entropy and Information Gain."
)


def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    ent = 0

    for c in counts:
        p = c / np.sum(counts)
        ent -= p * math.log2(p)

    return ent


def information_gain(data, feature, target):
    total_entropy = entropy(data[target])

    values, counts = np.unique(data[feature], return_counts=True)

    weighted_entropy = 0

    for i in range(len(values)):
        subset = data[data[feature] == values[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset[target])

    return total_entropy - weighted_entropy

st.sidebar.header("Dataset Source")

data_option = st.sidebar.selectbox(
    "Select dataset option",
    ["Upload CSV file", "Use synthetic dataset"]
)



if data_option == "Upload CSV file":

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is None:
        st.warning("Please upload a CSV file.")
        st.stop()

    data = pd.read_csv(uploaded_file)

else:
    # Synthetic dataset
    data = pd.DataFrame({
        "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast",
                    "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Mild",
                        "Cool", "Mild", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal",
                     "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
                 "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
        "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                        "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    })



st.subheader("Dataset Preview")
st.dataframe(data)



st.sidebar.header("Target Selection")

target_col = st.sidebar.selectbox(
    "Select Target Label",
    data.columns
)


features = [col for col in data.columns if col != target_col]



st.subheader("Feature Analysis (ID3)")

base_entropy = entropy(data[target_col])

st.write("Target Entropy :", round(base_entropy, 4))



gains = {}

for feat in features:
    gains[feat] = information_gain(data, feat, target_col)



gain_df = pd.DataFrame({
    "Feature": list(gains.keys()),
    "Information Gain": list(gains.values())
}).sort_values(by="Information Gain", ascending=False)

st.subheader("Information Gain of Features")
st.dataframe(gain_df)



best_feature = gain_df.iloc[0]["Feature"]

st.success(f"Best Feature for first split (ID3) : {best_feature}")


st.subheader("Feature Importance (Information Gain)")

fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    gain_df["Feature"],
    gain_df["Information Gain"]
)

ax.set_xlabel("Features")
ax.set_ylabel("Information Gain")
ax.set_title("ID3 – Feature Prioritization")

plt.xticks(rotation=45)

st.pyplot(fig)



st.subheader("Target Variable Distribution")

fig2, ax2 = plt.subplots()

data[target_col].value_counts().plot(
    kind="bar",
    ax=ax2
)

ax2.set_title("Target Variable Distribution")
ax2.set_xlabel("Class")
ax2.set_ylabel("Count")

st.pyplot(fig2)



st.info("ID3 algorithm selects the feature with maximum Information Gain at each node.")
