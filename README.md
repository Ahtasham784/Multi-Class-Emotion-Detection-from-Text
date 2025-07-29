# Multi-Class-Emotion-Detection-from-Text
# install libraries
!pip install datasets gradio scikit-learn nltk matplotlib
# import libraries
import re
import nltk
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# download stopwords
nltk.download('stopwords')
stop = set(stopwords.words("english"))
# load dataset
dataset = load_dataset("dair-ai/emotion")
train = dataset["train"]
labels = train.features["label"].names  # ['sadness','joy','love','anger','fear','surprise']
# preprocess text
def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^a-z\s]", "", s)
    tokens = [w for w in s.split() if w not in stop]
    return " ".join(tokens)

train = train.map(lambda x: {"clean": clean_text(x["text"])})
# train model(TF-IDF and Logistic Regression)
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train["clean"])
y_train = train["label"]

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
# emoji mapping
label_map = {
    "sadness": "😢 Sad",
    "joy": "😊 Happy",
    "love": "❤️ Love",
    "anger": "😠 Angry",
    "fear": "😱 Fear",
    "surprise": "😮 Surprise"
}
# prediction function
def predict_emotion(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    probs = model.predict_proba(X)[0]
    idx = np.argmax(probs)
    predicted_label = labels[idx]
    
    # Bar chart using matplotlib
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(labels, probs, color="skyblue")
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence per Emotion")
    plt.tight_layout()
    
    return f"{label_map[predicted_label]}", fig
# gradio GUI
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence..."),
    outputs=[gr.Label(label="Predicted Emotion"), gr.Plot(label="Confidence Chart")],
    title="Multi-Class Emotion Detection",
    description="Enter a sentence and the model will predict the emotion (joy, sadness, anger, fear, love, surprise)."
)

iface.launch()
