import streamlit as st
import numpy as np
import pickle
import pandas as pd
from final import vectorize

#fetching the model
model= pickle.load(open('modelfile.pkl', 'rb'))

#function to use the model
def classify(inp):
    
    inp= vectorize([inp])
    inp=inp.toarray()
    output=(model.predict(inp)) 
    return output   

negative_keywords = pd.read_csv('profanity_en.csv')
negative_keywords=negative_keywords[["text"]]

def contains_negative_sentiment(text):
    # Check if the text expresses negative sentiment
    for keyword in negative_keywords["text"]:
        if keyword in text:
            return np.array(" (Contains Abusive Language)")
    return np.array("")
    
def main():
    st.title("Hate Speech Detection \n By Yuvika Gupta")
    html_temp=""" 
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Hate Speech Detection </h2>
    </div>"""
    st.markdown(html_temp, unsafe_allow_html=True)
    
    tweet= st.text_input("Enter input","Type here")
    
    
    if st.button("Detect"):
        output1=classify(tweet)
        output2=contains_negative_sentiment(tweet)
        output= output1 + output2
        st.success('The text is classified as {}'.format(output))
        
         
if __name__=='__main__':
    main()
    
