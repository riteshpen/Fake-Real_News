import streamlit as st
import joblib
import spacy

# Load the model
model = joblib.load('code/grad_boost.pkl')

# Load the spaCy model
nlp = spacy.load("/opt/anaconda3/lib/python3.11/site-packages/en_core_web_lg/en_core_web_lg-3.7.1")

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def predict_fake_news(text):
    preprocessed_text = preprocess_text(text)
    vector = nlp(preprocessed_text).vector
    prediction = model.predict([vector])
    return "Fake News" if prediction == 1 else "Real News"

st.title("Fake News Detection")
user_input = st.text_area("Enter News Article")

if st.button("Predict"):
    result = predict_fake_news(user_input)
    st.write(result)
