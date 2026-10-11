
"""
app.py
------
Streamlit UI for Fake News Detection.

Models:
1. Linear SVM + TF-IDF
2. BiLSTM Deep Learning Model
3. DistilBERT Transformer Model

Run from project root:
    streamlit run app/app.py

Run from inside the app folder:
    streamlit run app.py
"""

import os
import sys

import streamlit as st
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)

from predict import predict_news
from predict_bilstm import predict_news_bilstm
from predict_bert import predict_news_bert


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------
# CUSTOM CSS — DARK DASHBOARD THEME
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Remove Streamlit's white top header strip */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
    }

    /* Remove the toolbar space */
    div[data-testid="stToolbar"] {
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Main application background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0b1020 0%,
            #151b35 50%,
            #202044 100%
        );
        color: #f1f5f9;
    }

    /* Page width and spacing */
    .block-container {
        max-width: 1500px;
        padding-top: 2rem !important;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Project title */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.25;
        margin-top: 0.2rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(
            90deg,
            #67e8f9,
            #a78bfa
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: #e0e7ff !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
    }

    [data-testid="stMetricValue"] {
        color: #67e8f9 !important;
        font-size: 1.5rem;
    }

    /* News article input */
    .stTextArea textarea {
        background-color: #111827;
        color: #f8fafc;
        border: 1px solid #6366f1;
        border-radius: 12px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(
            90deg,
            #6366f1,
            #8b5cf6
        );
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed
        );
        border: 1px solid #a5b4fc;
        color: white;
    }

    /* FAKE result card */
    .result-fake {
        background-color: rgba(239, 68, 68, 0.13);
        border-left: 5px solid #f87171;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        min-height: 135px;
    }

    /* REAL result card */
    .result-real {
        background-color: rgba(34, 197, 94, 0.13);
        border-left: 5px solid #4ade80;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        min-height: 135px;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Dividers */
    hr {
        border-color: rgba(148, 163, 184, 0.25);
    }

    /* Footer */
    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '📰 Fake News Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An intelligent news analysis dashboard powered by '
    'NLP, Machine Learning and Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    "Analyze a news article using three different prediction models."
)


# ---------------------------------------------------------
# INPUT
# ---------------------------------------------------------

st.markdown("###  Analyze a News Article")

news_text = st.text_area(
    "Enter a news article or headline:",
    height=150,
    placeholder="Paste a headline or full article text here..."
)

button_col, info_col = st.columns([1, 4])

with button_col:
    check_clicked = st.button(
        " Check News",
        type="primary",
        use_container_width=True
    )

with info_col:
    st.caption(
        "All three models analyze the same input independently."
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if check_clicked:

    if not news_text.strip():

        st.warning("Please enter some text before checking.")

    else:

        svm_result = None
        bilstm_result = None
        bert_result = None

        # =================================================
        # RUN ALL THREE MODELS
        # =================================================

        with st.spinner("Analyzing the article with all models..."):

            try:
                svm_result = predict_news(news_text)
            except Exception as e:
                st.error(f"SVM model error: {e}")

            try:
                bilstm_result = predict_news_bilstm(news_text)
            except FileNotFoundError as e:
                st.error(f"BiLSTM artifacts not found: {e}")
            except Exception as e:
                st.error(f"BiLSTM model error: {e}")

            try:
                bert_result = predict_news_bert(news_text)
            except FileNotFoundError as e:
                st.error(f"DistilBERT artifacts not found: {e}")
            except Exception as e:
                st.error(f"DistilBERT model error: {e}")


        # =================================================
        # SIDE-BY-SIDE MODEL PREDICTION CARDS
        # =================================================

        st.divider()
        st.subheader(" Model Predictions")

        model_columns = st.columns(3, gap="medium")

        model_cards = [
            {
                "column": model_columns[0],
                "title": "Linear SVM + TF-IDF",
                "result": svm_result,
                "name": "SVM"
            },
            {
                "column": model_columns[1],
                "title": "BiLSTM Deep Learning",
                "result": bilstm_result,
                "name": "BiLSTM"
            },
            {
                "column": model_columns[2],
                "title": "DistilBERT Transformer",
                "result": bert_result,
                "name": "DistilBERT"
            }
        ]

        for card in model_cards:

            with card["column"]:

                st.markdown(f"#### {card['title']}")

                result = card["result"]

                if result is not None:

                    label = result["label"]
                    confidence = result["confidence"]

                    if label == "FAKE":
                        st.markdown(
                            '<div class="result-fake">',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="result-real">',
                            unsafe_allow_html=True
                        )

                    st.markdown(f"### {label} NEWS")
                    st.markdown(f"**Confidence:** {confidence}%")

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                    if card["name"] == "SVM":

                        st.caption("Model: Linear SVM + TF-IDF")

                        if not result["is_calibrated"]:
                            st.caption(
                                "Decision-score-based confidence; "
                                "not a calibrated probability."
                            )

                    elif card["name"] == "BiLSTM":

                        st.caption("Model: BiLSTM")

                        st.caption(
                            "Sigmoid output; not a calibrated "
                            "probability."
                        )

                    else:

                        st.caption(
                            f"Model: {result['model_used']}"
                        )

                        st.caption(
                            "Softmax output; not a calibrated "
                            "probability."
                        )

                else:

                    st.warning(
                        f"{card['name']} prediction is unavailable."
                    )


        # =================================================
        # MODEL COMPARISON AND AGREEMENT
        # =================================================

        if (
            svm_result is not None
            and bilstm_result is not None
            and bert_result is not None
        ):

            st.divider()
            st.subheader(" Model Agreement")

            svm_label = svm_result["label"]
            bilstm_label = bilstm_result["label"]
            bert_label = bert_result["label"]

            agreement_col1, agreement_col2, agreement_col3 = (
                st.columns(3)
            )

            with agreement_col1:
                st.metric("SVM Prediction", svm_label)

            with agreement_col2:
                st.metric("BiLSTM Prediction", bilstm_label)

            with agreement_col3:
                st.metric("DistilBERT Prediction", bert_label)

            if svm_label == bilstm_label == bert_label:

                st.success(
                    f"✓ All three models predict **{svm_label} NEWS**"
                )

            else:

                st.warning(
                    "⚠️ The models produced different predictions. "
                    "Consider checking independent sources."
                )


        # =================================================
        # VISUAL ANALYTICS: THREE PIE CHARTS
        # =================================================

        available_results = [
            ("SVM + TF-IDF", svm_result),
            ("BiLSTM", bilstm_result),
            ("DistilBERT", bert_result)
        ]

        available_results = [
            (name, result)
            for name, result in available_results
            if result is not None
        ]

        if available_results:

            st.divider()
            st.subheader(" Prediction Score Analytics")

            st.caption(
                "Each chart shows the model's reported confidence "
                "and the remainder to 100%. The remainder is a "
                "visual complement, not a measured probability "
                "of the opposite class."
            )

            chart_columns = st.columns(
                len(available_results),
                gap="medium"
            )

            for index, (model_name, result) in enumerate(
                available_results
            ):

                with chart_columns[index]:

                    label = result["label"]

                    confidence = float(result["confidence"])
                    confidence = max(0.0, min(100.0, confidence))

                    remainder = 100.0 - confidence

                    st.markdown(f"#### {model_name}")
                    st.markdown(f"**Prediction: {label}**")

                    if label == "FAKE":
                        confidence_color = "#f87171"
                    else:
                        confidence_color = "#4ade80"

                    remainder_color = "#475569"

                    fig, ax = plt.subplots(figsize=(4, 3.5))

                    values = [confidence, remainder]
                    colors = [
                        confidence_color,
                        remainder_color
                    ]

                    if remainder <= 0:
                        values = [100]
                        colors = [confidence_color]

                    ax.pie(
                        values,
                        colors=colors,
                        startangle=90,
                        counterclock=False,
                        autopct=lambda pct: (
                            f"{pct:.1f}%" if pct >= 5 else ""
                        ),
                        pctdistance=0.68,
                        wedgeprops={
                            "width": 0.38,
                            "edgecolor": "#151b35",
                            "linewidth": 2
                        },
                        textprops={
                            "color": "white",
                            "fontsize": 10,
                            "weight": "bold"
                        }
                    )

                    ax.text(
                        0,
                        0,
                        f"{confidence:.1f}%",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=17,
                        fontweight="bold"
                    )

                    fig.patch.set_facecolor("#151b35")
                    ax.set_facecolor("#151b35")

                    ax.set_title(
                        "Reported Confidence",
                        color="white",
                        fontsize=11,
                        pad=12
                    )

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                    plt.close(fig)

                    st.markdown(
                        f"""
                        <div style="
                            display:flex;
                            justify-content:center;
                            gap:12px;
                            flex-wrap:wrap;
                            font-size:0.85rem;
                            color:#e2e8f0;
                        ">
                            <span>
                                <span style="color:{confidence_color};">
                                    ●
                                </span>
                                Score: {confidence:.1f}%
                            </span>
                            <span>
                                <span style="color:#64748b;">
                                    ●
                                </span>
                                Remainder: {remainder:.1f}%
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        # =================================================
        # QUICK SUMMARY AND DETAILED RESULTS
        # =================================================

        if available_results:

            st.divider()

            summary_col, details_col = st.columns(
                [1, 2],
                gap="large"
            )

            prediction_labels = [
                result["label"]
                for name, result in available_results
            ]

            fake_count = prediction_labels.count("FAKE")
            real_count = prediction_labels.count("REAL")

            with summary_col:

                st.subheader(" Quick Summary")

                st.metric(
                    "Models Analyzed",
                    f"{len(available_results)} / 3"
                )

                st.metric(
                    "FAKE Predictions",
                    fake_count
                )

                st.metric(
                    "REAL Predictions",
                    real_count
                )

                if len(set(prediction_labels)) == 1:

                    st.success(
                        f"All available models agree: "
                        f"{prediction_labels[0]}"
                    )

                else:

                    st.warning(
                        "Prediction disagreement detected."
                    )

            with details_col:

                st.subheader(" Detailed Model Results")

                with st.expander(
                    "View detailed model results",
                    expanded=False
                ):

                    for model_name, result in available_results:

                        st.markdown(f"**{model_name}**")
                        st.write(
                            f"Prediction: {result['label']}"
                        )
                        st.write(
                            f"Reported confidence: "
                            f"{result['confidence']}%"
                        )
                        st.divider()


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

footer_col1, footer_col2 = st.columns([3, 1])

with footer_col1:

    st.caption(
        "Academic Project | Fake News Detection Using NLP, "
        "Machine Learning and Deep Learning"
    )

with footer_col2:

    st.caption("SVM • BiLSTM • DistilBERT")

st.caption(
    "Model predictions are statistical signals and should not "
    "be treated as definitive fact-checking."
)
