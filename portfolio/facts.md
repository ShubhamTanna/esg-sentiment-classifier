# Facts: ESG Sentiment Classifier

| Fact | Value | Source |
|---|---|---|
| Headline | 2 of 5 real-world sentences right, after 86.84% accuracy on the test set | notebook cells 5 and 11 |
| F1 negative / positive | 0.866 / 0.871 | notebook cell 5 output |
| Data | 910 sentences: 471 positive, 439 negative | `data/ESG_Sentiment.csv` |
| Split | 682 train / 228 test, stratified, random_state 42 | notebook cell 5 |
| Confusion matrix | 97 / 13 / 17 / 101 (TN / FP / FN / TP) | derived from the classification report: recall 0.882 × 110 and 0.856 × 118 |
| Train/test overlap | 16 test sentences also appear in training | README |
| Context | Group project, Semester 7, 2025; I built the SVM, one of the models the group compared; a teammate prepared the dataset | README |
| Demo: "Stakeholders expressed disappointment with the board's lack of diversity." | Negative, margin −0.58 (human label Negative) | `app/streamlit_app.py` |
| Demo: "The company's carbon footprint remains a significant concern for investors." | Positive, margin +0.59 (human label Negative) | `app/streamlit_app.py` |

## Caveats

- The training sentences follow a few templates, so the model partly learned template words rather than sentiment. That's why it does far worse on real-world text.
