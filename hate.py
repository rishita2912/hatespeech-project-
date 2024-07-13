# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:04:54 2023

@author: RISHITA
"""

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
nltk.download('stopwords')
from nltk.util import pr
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words("english"))
import streamlit as st
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression

# Load the trained model and vectorizer
#model = LogisticRegression()
#vectorizer = TfidfVectorizer()

# Function to predict hate speech
def predict_hate_speech(text):
    test_data = text
    df = pd.read_csv("C:\Users\RISHITA\Desktop\twitter_data")
    df['labels']=df['class'].map({0:"Hate Speech Detected", 1:"Offensive language detected", 2:"NO hate and Offensive language detected"})
    df = df[['tweet','labels']]
    def clean(text):
      text = str(text).lower()
      text = re.sub('\[.*?\]', '',text)
      text = re.sub('https?://\S+|www\.\S', '', text)
      text = re.sub('<.*?>+', '', text)
      text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
      text = re.sub('\n', '', text)
      text = re.sub('\w*\d\w*', '', text)
      text = [word for word in text.split(' ') if word not in stopword]
      text =" ".join(text)
      text = [stemmer.stem(word) for word in text.split(' ')]
      text = " ".join(text)
      return text
    df["tweet"] = df["tweet"].apply(clean)
    #print(df.head())
    x = np.array(df["tweet"])
    y = np.array(df["labels"])
    cv = CountVectorizer()
    x = cv.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size= 0.33, random_state= 42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    df = cv.transform([test_data]).toarray()
    return (clf.predict(df))

# Set the app title
st.title("Hate Speech Detection")

# Add a text input field
text_input = st.text_input("Enter a text:")

# Add a button to trigger the prediction
if st.button("Predict"):
    if text_input:
        prediction = predict_hate_speech(text_input)
        st.write("Prediction:", prediction)
    else:
        st.write("Please enter a text.")
