import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index={value: key for key, value in word_index.items()}

# Load pretrained model
model=load_model('simple_rnn_imdb.h5')

#Step2: Helper function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review ])

## Function to preprocess
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+ 3 for word in words]
    padded_review= sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# Step 3: Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]

# Streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).")

user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=   preprocess_text(user_input)
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f"Predicted Sentiment: {sentiment} (Confidence: {prediction[0][0]:.4f})")
else:
    st.write("Please enter a movie review and click 'Classify' to see the sentiment prediction.")