import streamlit as st
import spacy
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load the spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("/opt/anaconda3/lib/python3.11/site-packages/en_core_web_lg/en_core_web_lg-3.7.1")

nlp = load_spacy_model()

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# Load pre-trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('gradient_boosting_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Function to make a prediction
def predict_fake_news(headline):
    # Preprocess the input headline
    tokens = preprocess_text(headline)
    preprocessed_text = ' '.join(tokens)
    
    # Convert to vector
    vector = nlp(preprocessed_text).vector.reshape(1, -1)
    
    # Scale the vector 
    vector_scaled = scaler.transform(vector)
    
    # Predict using the trained model
    prediction = model.predict(vector_scaled)
    prediction_prob = model.predict_proba(vector_scaled)
    
    # Extract the confidence for the prediction
    confidence = prediction_prob[0][1] if prediction[0] == 1 else prediction_prob[0][0]
    
    return ('Real News' if prediction[0] == 1 else 'Fake News', confidence)

# Streamlit app
st.title("Fake News Detection")
st.write("Enter a news headline to check if it is real or fake.")

# Input from the user
headline = st.text_input("News Headline", "")

# Display prediction when user enters a headline
if headline:
    # Predict the result
    result, confidence = predict_fake_news(headline)
    st.write(f"The news is likely: **{result}** with confidence: **{confidence:.2f}**")

    # Plot Word Frequency Distribution
    tokens = preprocess_text(headline)
    token_counts = Counter(tokens)
    df_token_counts = pd.DataFrame(token_counts.items(), columns=['Token', 'Count']).sort_values(by='Count', ascending=False)
    
    st.write("### Word Frequency Distribution")
    st.bar_chart(df_token_counts.set_index('Token'))

    # Display the vector representation
    st.write("### Headline Vector Representation")
    vector = nlp(' '.join(tokens)).vector.reshape(1, -1)
    df_vector = pd.DataFrame(vector, columns=[f'Feature {i}' for i in range(vector.shape[1])])
    st.write(df_vector)
    
    # Display a table with scaled features
    vector_scaled = scaler.transform(vector)
    df_vector_scaled = pd.DataFrame(vector_scaled, columns=[f'Scaled Feature {i}' for i in range(vector_scaled.shape[1])])
    st.write("### Scaled Feature Vector")
    st.write(df_vector_scaled)
