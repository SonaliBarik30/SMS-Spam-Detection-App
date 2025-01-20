import streamlit as st
import pickle

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))

st.title("SMS Spam Classification")
st.write("This is a machine learning project to classify the sms are spam or not")
user_input=st.text_area("Enter the sms to classify",height=150)

if st.button("Classify"):
    if user_input:
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)
        if result[0] == 1:
            st.write("This is a Spam")
        else :
            st.write("This is not a Spam")
    else :
        st.write("Please type the sms to classify") 