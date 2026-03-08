import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re


def pad_sequences(sequences, maxlen, padding="pre", truncating="pre", value=0):
    result = []
    for seq in sequences:
        seq = list(seq)
        if len(seq) > maxlen:
            seq = seq[-maxlen:] if truncating == "pre" else seq[:maxlen]
        pad_len = maxlen - len(seq)
        pad = [value] * pad_len
        seq = pad + seq if padding == "pre" else seq + pad
        result.append(seq)
    return np.array(result, dtype=np.int32)


st.set_page_config(
    page_title="CineScope · Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background-color: #0d0d0d;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(220,80,50,0.08) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, rgba(255,180,50,0.06) 0%, transparent 60%);
    color: #f0ece4;
}

h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.8rem !important;
    letter-spacing: -0.5px;
    color: #f0ece4 !important;
    margin-bottom: 0 !important;
}

h2, h3 { font-family: 'Playfair Display', serif !important; color: #f0ece4 !important; }

.stTextArea textarea {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: #f0ece4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    caret-color: #dc5032;
}

.stTextArea textarea:focus {
    border-color: #dc5032 !important;
    box-shadow: 0 0 0 2px rgba(220,80,50,0.25) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #dc5032, #c0392b) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.result-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}

.sentiment-label { font-family: 'Playfair Display', serif; font-size: 2rem; margin-bottom: 0.25rem; }
.sentiment-positive { color: #f5c518; }
.sentiment-negative { color: #dc5032; }

.confidence-text { font-size: 1rem; color: #888; margin-bottom: 0.5rem; }

.conf-bar-bg { background: #2a2a2a; border-radius: 99px; height: 8px; width: 100%; margin-top: 0.6rem; }
.conf-bar-fill-pos { background: linear-gradient(90deg, #b8860b, #f5c518); height: 8px; border-radius: 99px; }
.conf-bar-fill-neg { background: linear-gradient(90deg, #8b1a0a, #dc5032); height: 8px; border-radius: 99px; }

.section-divider { border: none; border-top: 1px solid #222; margin: 1.5rem 0; }

.metrics-row { display: flex; gap: 1rem; margin: 1rem 0; }
.metric-box {
    flex: 1;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 1.5rem; font-weight: 600; margin-top: 4px; }

.footer { text-align: center; color: #444; font-size: 0.75rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    import tensorflow as tf
    model = tf.keras.models.load_model("sentiment_rnn.keras")

    return model, tokenizer


def preprocess(text, tokenizer, max_length=200):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_length)


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def make_probability_chart(pos_prob):
    neg_prob = 1 - pos_prob
    labels = ["Negative", "Positive"]
    probs = [neg_prob, pos_prob]
    colors = ["#dc5032", "#f5c518"]

    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    bars = ax.barh(labels, probs, color=colors, height=0.45, edgecolor="none")

    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{prob*100:.1f}%",
            va="center", ha="left",
            color="#f0ece4", fontsize=11, fontweight="500",
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Probability", color="#555", fontsize=9)
    ax.tick_params(colors="#888", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#2a2a2a", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def make_wordcloud(text, sentiment):
    color = "#f5c518" if sentiment == "positive" else "#dc5032"

    def single_color_func(*args, **kwargs):
        return color

    wc = WordCloud(
        width=800, height=350,
        background_color="#1a1a1a",
        color_func=single_color_func,
        prefer_horizontal=0.85,
        max_words=80,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


st.markdown("# 🎬 CineScope")
st.markdown(
    "<p style='color:#666; font-size:1rem; margin-top:-0.5rem;'>"
    "Paste a movie review and discover its sentiment instantly."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

try:
    model, tokenizer = load_artifacts()
except Exception as e:
    st.markdown("""
    <div style="background:#1a0a0a;border:1px solid #5a1a1a;border-radius:12px;padding:1.5rem 2rem;margin:1rem 0;">
        <div style="color:#dc5032;font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;">
            ⚠️ Model files not found
        </div>
        <div style="color:#aaa;font-size:0.9rem;line-height:1.7;">
            Run the <b>movie.ipynb</b> notebook first to train and save the model.<br>
            It will generate <code>sentiment_rnn.keras</code> and <code>tokenizer.pkl</code> in the same folder.
        </div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("Full error details"):
        st.code(str(e), language="text")
    st.stop()

MAX_LENGTH = 200

review = st.text_area(
    "Movie Review",
    placeholder='e.g. "An absolute masterpiece. The direction was flawless and the performances deeply moving..."',
    height=160,
    label_visibility="collapsed",
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    analyze = st.button("Analyze →", use_container_width=True)
with col_info:
    if review:
        wc = word_count(review)
        st.markdown(
            f"<p style='color:#555; font-size:0.85rem; padding-top:0.6rem;'>"
            f"{wc} word{'s' if wc != 1 else ''} · {len(review)} characters</p>",
            unsafe_allow_html=True,
        )

if analyze:
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        with st.spinner("Analyzing sentiment…"):
            padded = preprocess(review, tokenizer, MAX_LENGTH)
            raw_score = float(model.predict(padded, verbose=0)[0][0])

        is_positive = raw_score > 0.5
        confidence = raw_score if is_positive else 1 - raw_score
        sentiment = "positive" if is_positive else "negative"
        label = "Positive 😀" if is_positive else "Negative 😞"
        color_cls = "sentiment-positive" if is_positive else "sentiment-negative"
        bar_cls = "conf-bar-fill-pos" if is_positive else "conf-bar-fill-neg"
        conf_pct = round(confidence * 100, 1)

        st.markdown(f"""
        <div class="result-card">
            <div class="sentiment-label {color_cls}">{label}</div>
            <div class="confidence-text">Confidence score</div>
            <div class="conf-bar-bg">
                <div class="{bar_cls}" style="width:{conf_pct}%"></div>
            </div>
            <p style="font-size:0.85rem; color:#666; margin-top:0.4rem;">{conf_pct}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-label">Raw Score</div>
                <div class="metric-value" style="color:#888;">{raw_score:.4f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Positive Prob.</div>
                <div class="metric-value" style="color:#f5c518;">{raw_score*100:.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Negative Prob.</div>
                <div class="metric-value" style="color:#dc5032;">{(1-raw_score)*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        st.markdown("### Sentiment Probability")
        st.pyplot(make_probability_chart(raw_score), use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        st.markdown("### Key Words in Your Review")
        st.pyplot(make_wordcloud(review, sentiment), use_container_width=True)

st.markdown(
    "<div class='footer'>CineScope · Powered by RNN Sentiment Analysis</div>",
    unsafe_allow_html=True,
)
