import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

def transform_text(text):
    #lowercase
    text = text.lower()
    #tokenization
    text = nltk.word_tokenize(text)

    processed_text=[]
    #removing special characters
    for i in text:
        if i.isalnum():
            processed_text.append(i)

    text=processed_text[:]
    processed_text.clear()

    #removing stop words
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            processed_text.append(i)

    text = processed_text[:]
    processed_text.clear()

    ps = PorterStemmer()
    #stemming
    for i in text:
        processed_text.append(ps.stem(i))
    return " ".join(processed_text)

# Loading the pre-trained model and vectorizer
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# App interface
def main():
    st.title("Email/SMS Spam Classifier")
    st.write("This application predicts whether a message is spam or not using a machine learning model.")

    # User input
    user_input = st.text_area("Enter your message here:")
    
    # Prediction
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message to classify.")
        else:
            model, vectorizer = load_model()
            transformed_input = transform_text(user_input)
            vector_input = vectorizer.transform([transformed_input])
            prediction = model.predict(vector_input)[0]
            prediction_proba = model.predict_proba(vector_input)[0]

            # Display results
            if prediction== 1:
                st.error("The message is classified as **SPAM**.")
            else:
                st.success("The message is classified as **NOT SPAM**.")

            # Display probabilities
            st.write("Confidence:")
            st.write(f"Spam: {prediction_proba[1]*100:.2f}%")
            st.write(f"Not Spam: {prediction_proba[0]*100:.2f}%")

if __name__ == "__main__":
    main()
