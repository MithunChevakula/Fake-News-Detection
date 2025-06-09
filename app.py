import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("🕵️‍♂️Fake News Detector")
st.write("Enter a news article below to check whether it is Fake or Real.")

news_input = st.text_area("News Article:", "")

if st.button("🔍Check News"):
    if news_input.strip():

        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✅ The News is Real!")
        else:
            st.error("❌ This news is Fake.")
    else:
        st.warning("⚠️ Please enter some text to analyze.")