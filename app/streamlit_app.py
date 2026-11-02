"""
ESG Sentiment Classifier: live demo.

Type a sentence about a company's environmental, social or governance practice and the TF-IDF + Linear SVM
model labels it positive or negative, showing which words pushed it each way. The model is retrained at
start-up with exactly the notebook's settings (75/25 stratified split, random_state=42), so the accuracy
shown matches the notebook: 86.84%.

Run locally:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA = Path(__file__).resolve().parents[1] / "data" / "ESG_Sentiment.csv"
EXAMPLES = [  # the five real-world sentences tested in the notebook, with the human label
    ("The company announced a new plan to achieve net-zero emissions by 2040.", "Positive"),
    ("Regulators are investigating the firm over alleged human rights violations in its supply chain.", "Negative"),
    ("The foundation's investment in renewable energy has yielded strong community support.", "Positive"),
    ("Stakeholders expressed disappointment with the board's lack of diversity.", "Negative"),
    ("The company's carbon footprint remains a significant concern for investors.", "Negative"),
]

st.set_page_config(page_title="ESG Sentiment Classifier", layout="centered")


@st.cache_resource
def train():
    df = pd.read_csv(DATA)
    df["Sentiment_Label"] = df["Sentiment"].map({"Positive": 1, "Negative": 0})
    X_tr, X_te, y_tr, y_te = train_test_split(df["Sentence"], df["Sentiment_Label"], test_size=0.25,
                                              random_state=42, stratify=df["Sentiment_Label"])
    pipe = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
                     ("model", LinearSVC(random_state=42))])
    pipe.fit(X_tr, y_tr)
    return pipe, accuracy_score(y_te, pipe.predict(X_te)), len(X_tr), len(X_te)


def explain(pipe, text, k=6):
    tfidf, svm = pipe.named_steps["tfidf"], pipe.named_steps["model"]
    vec = tfidf.transform([text])
    contrib = vec.multiply(svm.coef_[0]).tocoo()
    vocab = np.array(tfidf.get_feature_names_out())
    words = pd.DataFrame({"word": vocab[contrib.col], "weight": contrib.data})
    return words.sort_values("weight", key=abs, ascending=False).head(k)


pipe, acc, n_tr, n_te = train()

st.title("ESG sentiment classifier")
st.caption(f"TF-IDF + Linear SVM · trained on {n_tr} labelled ESG sentences · "
           f"{acc:.2%} accuracy on {n_te} held-out sentences")

pick = st.selectbox("Try a real-world sentence (not in the training data), or write your own",
                    [EXAMPLES[3][0]] + [s for i, (s, _) in enumerate(EXAMPLES) if i != 3] + ["Write my own"])
text = st.text_area("Sentence", value="" if pick == "Write my own" else pick, height=90,
                    placeholder="e.g. The company cut its water use by 30% across all plants.")

if text.strip():
    score = pipe.decision_function([text])[0]
    label = "Positive" if score > 0 else "Negative"
    c1, c2 = st.columns(2)
    c1.metric("Predicted sentiment", label)
    c2.metric("Margin (distance from the boundary)", f"{score:+.2f}")
    truth = dict(EXAMPLES).get(text.strip())
    if truth:
        (st.success if truth == label else st.error)(f"Human label: {truth}. "
                                                     f"The model is {'right' if truth == label else 'wrong'} here.")
    words = explain(pipe, text)
    if words.empty:
        st.warning("None of these words appear in the training vocabulary, so the model is guessing.")
    else:
        st.write("**Words that drove the call** (positive weight pushes towards Positive):")
        st.bar_chart(words.set_index("word")["weight"], horizontal=True)

with st.expander("Why it gets real-world sentences wrong"):
    st.markdown(
        "On the five real-world sentences above, the model gets only **2 of 5** right. The training sentences "
        "follow a few templates, so the model partly learned template words rather than sentiment. "
        "Negation and context (\"remains a significant **concern**\") are invisible to a bag-of-words model. "
        "The next step is retraining on real sustainability reports and news, or using a transformer model."
    )
