import joblib
from underthesea import word_tokenize
import re
import pandas as pd
import streamlit as st


def wordopt(text):
    text = text.lower()
    text = re.sub('https?:\/\/.*[\r\n]*', ' ', text)
    text = re.sub('[^\w\s]', ' ', text)
    text = re.sub('\n', ' ', text)
    return text


def tokenize(sentence):
    return word_tokenize(sentence, format='word')


def getModel(model_type):
    if model_type == "Decision Tree":
        return joblib.load('models/DT_model.joblib')
    else:
        return joblib.load('models/NB_model.joblib')


def predict(text, model_type):
    vectorizer = joblib.load('models/vectorizer.joblib')
    return manual_testing(text, getModel(model_type), vectorizer)


@st.cache
def manual_testing(news, model, vectorizer):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    pred = model.predict(new_xv_test)
    return pred[0]


def checker(model_type, text):
    global model_checker
    global text_checker
    checker = True
    if model_type == "None":
        model_checker.warning("Please choose a model")
        checker = False
    if text == "":
        text_checker.warning("Please input your text")
        checker = False
    return checker


st.title("Fake News Detection")
choices = ["None", "Decision Tree", "NaiveBayes"]

model_holder = st.empty()
model_type = model_holder.selectbox("Choose model", choices, index=0, key="model")
model_checker = st.empty()

text = st.text_area("Input your text here", "")
text_checker = st.empty()
if st.button("Predict now"):
    if checker(model_type, text):
        result = predict(text, model_type)
        if result == 1:
            st.error("This is fake news")
        else:
            st.success("This is not fake news")
