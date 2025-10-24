from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ✅ আরও বড় ট্রেনিং ডেটাসেট
texts = [
    "I love this product", "This is amazing", "I like this phone", "It is very good",
    "I hate this", "This is terrible", "I dislike it", "It is very bad", "I am happy", "I am sad"
]
labels = [
    'positive', 'positive', 'positive', 'positive',
    'negative', 'negative', 'negative', 'negative', 'positive', 'negative'
]

# Create text vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

print("✅ Model trained successfully! Now test it below 👇")

# Interactive loop
while True:
    text = input("\nEnter a sentence to classify (or 'exit'): ")
    if text.lower() == 'exit':
        break
    X_test = vectorizer.transform([text])
    prediction = model.predict(X_test)
    print("Prediction:", prediction[0])

