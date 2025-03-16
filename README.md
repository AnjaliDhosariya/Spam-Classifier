
# 📧 SMS & Email Spam Classifier

This is a **machine learning-based** web application that classifies messages as **Spam or Not Spam**. It is built using **Streamlit**, and the model is trained using **Natural Language Processing (NLP) techniques**.

## 🚀 Features

- **Real-time Spam Detection:** Classifies SMS or Email messages as spam or not spam.
- **NLP Preprocessing:** Tokenization, stopword removal, stemming.
- **Machine Learning Model:** Uses a trained model stored in `model.pkl`.
- **Web App Interface:** Built with **Streamlit** for easy interaction.

## 📂 Project Structure

```
📂 SMS-Spam-Detection
│── app.py                 # Streamlit app for spam classification
│── model.pkl              # Trained machine learning model
│── vectorizer.pkl         # TF-IDF Vectorizer for text transformation
│── sms-spam-detection.ipynb # Jupyter Notebook for training the model
│── spam.csv               # Dataset used for training
│── README.md              # Project documentation
```

## 🧠 How It Works

1. **User Input:** The user enters a message in the text box.
2. **Text Processing:** The message is tokenized, stopwords are removed, and stemming is applied.
3. **Vectorization:** The processed text is converted into numerical features using `vectorizer.pkl`.
4. **Prediction:** The trained `model.pkl` predicts whether the message is spam or not.
5. **Result Display:** The app shows the classification along with confidence scores.

## 🗂 Dataset: `spam.csv`

The dataset used for training this model is **"spam.csv"**, which contains a collection of SMS messages labeled as **spam** or **ham** (not spam).  

### 📊 Dataset Overview:
- **Total Entries:** ~5,574 messages  
- **Columns:**
  - `label`: The classification of the message (spam or ham).  
  - `message`: The actual text content of the SMS.  

### 📈 Label Distribution:
- **Ham (Not Spam):** ~86.6% of messages  
- **Spam:** ~13.4% of messages  

### 🔍 Sample Data:

| label | message |
|-------|---------|
| ham   | "Hey, are we still meeting for lunch today?" |
| spam  | "Congratulations! You have won a free iPhone. Claim now!" |

### ⚡ Preprocessing Steps:
1. Convert text to **lowercase**.
2. Remove **special characters** and **stopwords**.
3. Apply **stemming** to reduce words to their root forms.
4. Convert text into numerical representation using **TF-IDF Vectorization**.

## 📌 Technologies Used

- **Python**
- **Streamlit** (for UI)
- **NLTK** (Natural Language Processing)
- **Scikit-learn** (Machine Learning)
- **Pandas & NumPy** (Data processing)

