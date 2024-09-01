import streamlit as st
import spacy
import joblib

# Load the spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("/opt/anaconda3/lib/python3.11/site-packages/en_core_web_lg/en_core_web_lg-3.7.1")


nlp = load_spacy_model()

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

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
    preprocessed_text = preprocess_text(headline)
    # Convert to vector
    vector = nlp(preprocessed_text).vector.reshape(1, -1)
    # Scale the vector 
    vector_scaled = scaler.transform(vector)
    # Predict using the trained model
    prediction = model.predict(vector_scaled)
    return 'Real News' if prediction[0] == 1 else 'Fake News'

# Streamlit app
st.title("Fake News Detection")

st.write("Enter a news headline to check if it is real or fake.")

# Input from the user
headline = st.text_input("News Headline", "")

# Display prediction when user enters a headline
if headline:
    # Predict the result
    result = predict_fake_news(headline)
    st.write(f"The news is likely: **{result}**")
