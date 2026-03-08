# 🎬 Movie Sentiment Analysis

A deep learning-powered web app that analyzes movie reviews and predicts their sentiment — telling you whether a review is **positive** or **negative** (and how strongly so). Built with a Recurrent Neural Network (RNN) and deployed on Streamlit.

---

## 🚀 Live Demo

Try it out on Streamlit: **[movie-sentiment](https://movie-sentiment-j6w8yapsna2avgvz8xhapu.streamlit.app/)(#)**

---

## 🧠 How It Works

1. User inputs a movie review (text)
2. The text is preprocessed and tokenized
3. A trained **RNN model** processes the sequence
4. The app outputs the **sentiment** (Positive / Negative) along with a confidence score

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | Recurrent Neural Network (RNN) |
| Notebook | Jupyter Notebook |
| Framework | TensorFlow / Keras |
| Deployment | Streamlit |
| Language | Python |

---

## 📁 Project Structure

```
movie-sentiment-analysis/
│
├── notebook/
│   └── movie_sentiment_analysis.ipynb   # Model training & experimentation
│
├── model/
│   └── sentiment_model.h5               # Saved trained model
│
├── app.py                               # Streamlit app
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## ⚙️ Getting Started (Run Locally)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/movie-sentiment-analysis.git
cd movie-sentiment-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📓 Jupyter Notebook

The notebook walks through the full ML pipeline:

- Data loading & exploration
- Text preprocessing & tokenization
- Building & training the RNN model
- Evaluating model performance
- Saving the model for deployment

To run it:
```bash
jupyter notebook notebook/movie_sentiment_analysis.ipynb
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 83.37% |
| Loss | 0.5087 |
| Dataset | IMDB Movie Reviews |

> Update the scores above after training your final model.

---

## 🎯 Features

- 🔍 Real-time sentiment prediction from any movie review text
- 📈 Confidence score displayed alongside the result
- 🧹 Automatic text preprocessing (lowercasing, punctuation removal, etc.)
- 💻 Clean and minimal Streamlit UI

---

## 📦 Requirements

```
streamlit
tensorflow
keras
numpy
pandas
scikit-learn
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🙌 Acknowledgements

- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) — Large Movie Review Dataset by Stanford AI
- [TensorFlow / Keras](https://www.tensorflow.org/) — Model building
- [Streamlit](https://streamlit.io/) — App deployment

---

## 📄 License

This project is licensed under the **MIT License**.
