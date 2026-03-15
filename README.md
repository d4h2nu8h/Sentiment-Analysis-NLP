# Sentiment Classification using NLP and Deep Learning

> A multi-class text classification system that identifies the emotional polarity of written text — positive, negative, or neutral — using classical NLP models and a Long Short-Term Memory (LSTM) neural network, deployed as a live web application.

---

## Overview

Understanding public sentiment at scale is a critical capability for businesses, researchers, and product teams. Whether analysing customer reviews, social media posts, or survey responses, the ability to automatically detect emotional tone removes a significant bottleneck in qualitative data analysis.

This project builds a supervised sentiment classification pipeline trained on the **IMDB movie review dataset**, framing the problem as a multi-class classification task across three sentiment categories: positive, negative, and neutral. The system evaluates multiple approaches — from traditional Naive Bayes and Support Vector Machine baselines to a deep learning LSTM model — and surfaces predictions through a deployed web interface.

The project also explores fine-grained emotion detection, distinguishing between specific emotional states such as happiness, amusement, disappointment, and anger, beyond simple binary polarity classification.

---

## Dataset

**Source:** IMDB Movie Reviews Dataset

The dataset consists of labelled movie reviews widely used as a benchmark for natural language processing and sentiment analysis tasks. Each review is associated with a sentiment label (positive or negative), and neutral examples were constructed through preprocessing.

Key characteristics:

| Property | Detail |
|---|---|
| Domain | Movie reviews |
| Task type | Multi-class text classification |
| Classes | Positive, Negative, Neutral |
| Input format | Raw text (variable length) |

---

## Methodology

### Text Preprocessing

- Lowercasing, punctuation removal, and stop word filtering
- Tokenisation and sequence padding for LSTM compatibility
- TF-IDF vectorisation for classical ML models
- Train-test splitting with stratified sampling

### Models Trained

**Classical Machine Learning:**

1. Naive Bayes — fast probabilistic baseline using TF-IDF features
2. Support Vector Machine (SVM) — linear kernel with TF-IDF, strong baseline for text classification

**Deep Learning (`sentiment_analysis.h5`):**

3. Long Short-Term Memory (LSTM) — sequence model built with Keras, capable of capturing long-range dependencies in text

The LSTM architecture includes:
- Embedding layer trained end-to-end on the corpus vocabulary
- Stacked LSTM layers with dropout regularisation
- Dense output layer with softmax activation for multi-class prediction

### Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score (macro and weighted averages)
- Confusion matrix analysis across all three sentiment classes

---

## Results

The LSTM model outperformed the classical baselines on the held-out test set, particularly on longer and more ambiguous reviews where sequential context matters. SVM with TF-IDF features provided a strong, computationally lightweight alternative, performing competitively on shorter reviews.

> Full evaluation metrics and model comparisons are available in the source notebooks. The deployed Heroku application provides real-time predictions via the web interface.

---

## Limitations & Future Work

**Current Limitations:**

- The model is trained primarily on movie review language and may not generalise well to other domains (e.g., financial news, medical text) without fine-tuning
- Neutral class construction is heuristic-based, which may introduce label noise
- The LSTM model does not leverage pre-trained word embeddings (e.g., GloVe, Word2Vec), which limits representation quality on out-of-vocabulary terms
- Sarcasm and irony remain significant sources of misclassification across all models

**Future Directions:**

- Integrate transformer-based models (BERT, RoBERTa) for state-of-the-art contextual embeddings
- Expand to aspect-based sentiment analysis — detecting sentiment toward specific entities or topics within a review
- Fine-tune on domain-specific corpora to improve cross-domain generalisation
- Add confidence scores to web app predictions for better interpretability

---

## How to Run This Project

### Prerequisites

```bash
Python 3.8+
```

### 1. Clone the Repository

```bash
git clone https://github.com/d4h2nu8h/sentiment-classification-nlp.git
cd sentiment-classification-nlp
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Web Application Locally

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

### 4. Heroku Deployment

The app is configured for Heroku deployment via `Procfile` and `runtime.txt`. To deploy your own instance:

```bash
heroku create
git push heroku main
heroku open
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| NLP & ML | Scikit-learn, NLTK |
| Deep Learning | TensorFlow, Keras |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Deployment | Heroku |
| Data Format | Text / CSV |

---

## Author

**Dhanush Sambasivam**

[![GitHub](https://img.shields.io/badge/GitHub-d4h2nu8h-181717?style=flat&logo=github)](https://github.com/d4h2nu8h)

---

## License

This project is intended for academic and research purposes. The IMDB dataset is publicly available for non-commercial use.
