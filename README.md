# IMDB Sentiment Analysis using SimpleRNN

## Overview
This project performs sentiment analysis on movie reviews using a Simple Recurrent Neural Network (RNN). The model classifies reviews as positive or negative based on sequential patterns in text data.

The project demonstrates how deep learning models can be applied to Natural Language Processing (NLP) tasks using word embeddings and sequence modeling.

---

## Dataset
The IMDB dataset consists of:

- 50,000 movie reviews  
- Balanced positive and negative labels  

Each review is represented as a sequence of word indices.

---

## Text Preprocessing
The following steps were applied:

- Tokenization using predefined IMDB word index  
- Conversion of words to integer sequences  
- Padding sequences to fixed length (500)  
- Handling unknown and reserved tokens  

---

## Model Architecture
The model consists of:

- Embedding layer (word index → dense vector)  
- SimpleRNN layer for sequence modeling  
- Dense output layer with Sigmoid activation  

Loss Function: Binary Cross-Entropy  
Optimizer: Adam  

---

## Training Strategy
- Reviews padded to fixed length  
- Validation split for monitoring performance  
- Early Stopping to prevent overfitting  

---

## Inference Pipeline
A separate inference pipeline was built to:

- Accept raw English text  
- Apply same preprocessing as training  
- Predict sentiment with confidence score  

---

## Deployment
The trained model is deployed using Streamlit.  
Users can enter custom movie reviews and receive real-time sentiment predictions.

---

## Key Learnings
- How embeddings represent semantic meaning  
- How RNNs capture sequential dependencies  
- Importance of consistent preprocessing  
- Difference between NLP training and inference  
- End-to-end model deployment  

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Streamlit  
