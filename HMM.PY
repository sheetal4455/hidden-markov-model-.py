import streamlit as st
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

# Title
st.title("Hidden Markov Model Experiment")
st.write("Upload a dataset to analyze observation sequences using Hidden Markov Model")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    # Read dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data)

    # Select column
    column = st.selectbox("Select Observation Column", data.columns)

    observations = data[column].astype(str)

    # Encode observations
    encoder = LabelEncoder()
    encoded_obs = encoder.fit_transform(observations)

    X = encoded_obs.reshape(-1, 1)

    # Number of hidden states
    n_states = st.slider("Select number of Hidden States", 2, 5, 2)

    # Train HMM model
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=100)
    model.fit(X)

    # Predict hidden states
    hidden_states = model.predict(X)

    # Display results
    st.subheader("Encoded Observations")
    st.write(encoded_obs)

    st.subheader("Predicted Hidden States")
    st.write(hidden_states)

    st.subheader("Transition Matrix")
    st.write(model.transmat_)

    st.subheader("Emission Matrix")
    st.write(model.emissionprob_)

    st.success("Hidden Markov Model successfully trained and analyzed!")
