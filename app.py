import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
texts = [
    "I love this product", "This is amazing", "I like this phone", "It is very good",
    "I hate this", "This is terrible", "I dislike it", "It is very bad", "I am happy", "I am sad"
]
labels = [
    'positive', 'positive', 'positive', 'positive',
    'negative', 'negative', 'negative', 'negative', 'positive', 'negative'
]

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# Streamlit interface
st.title("Simple Text Classifier AI")
user_input = st.text_input("Enter a sentence to classify:")

if st.button("Predict"):
    X_test = vectorizer.transform([user_input])
    prediction = model.predict(X_test)
    st.success(f"Prediction: {prediction[0]}")
