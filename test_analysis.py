import streamlit as st
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
# Preprocess text (username and link placeholders)



@st.cache_resource()
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

model = AutoModelForSequenceClassification.from_pretrained(MODEL)

password_guess = st.text_input('What is the Password?',type="password") 
if password_guess != st.secrets["password"]: 
    st.stop()

st.title("Sentiment Analysis")
text = st.text_input("Enter text to analyze")

result = preprocess(text)
 
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
 
ranking = np.argsort(scores)
ranking = ranking[::-1]
if text:
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        st.write(f"{i+1}) {l} {np.round(float(s), 2)*100}%")

st.write("About me: https://www.linkedin.com/in/yesner-salgado/")
