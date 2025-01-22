import streamlit as st
import pickle

try:
    model = pickle.load(open('spam.pkl', 'rb'))
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please check the file paths.")
    st.stop()

# Apply custom CSS for black background
st.markdown("""
    <style>
    body {
        background-color: black;
        color: white;
    }
    .result-box {
        background-color: #FF6F61;  /* Pop-up color */
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        color: white;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI components
st.title("SMS Spam Classification")
st.write("This is a machine learning project to classify whether the SMS is spam or not.")

user_input = st.text_area("Enter the SMS to classify", height=150)

# Classify button
if st.button("Classify"):
    if user_input:
        # Transform the user input into the format expected by the model
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()

        # Make the prediction
        result = model.predict(vectorized_data)

        # Display the result with custom style for popup-like effect
        if result[0] == 1:
            st.markdown('<div class="result-box">🚫 This is a Spam message.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box">✅ This is not a Spam message.</div>', unsafe_allow_html=True)
    else:
        st.warning("Please type the SMS to classify.")
