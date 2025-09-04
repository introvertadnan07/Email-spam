# Only run this once to train and save your model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Example data
X = ["Hello there", "Win money now", "Important meeting", "Free prize"]
y = [0, 1, 0, 1]  # 0 = Not Spam, 1 = Spam

# Preprocessing function
def preprocess(text):
    # You can use your transform_text here
    return text.lower()

X = [preprocess(x) for x in X]

# Vectorize
tfidf = TfidfVectorizer()
X_vec = tfidf.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model and vectorizer
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))


import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ✅ Load trained objects
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])  # use transform, not fit_transform
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
