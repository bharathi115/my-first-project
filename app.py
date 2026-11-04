
import base64
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis | Machine Learning",
    page_icon="🩵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# SKY BLUE PROFESSIONAL THEME
# ---------------------------------------------------------

st.markdown("""
<style>

:root {
    --sky: #38bdf8;
    --blue: #0284c7;
    --deep: #075985;
    --pale: #eaf8ff;
    --border: #bae6fd;
}

.stApp {
    background: linear-gradient(
        135deg,
        #effaff 0%,
        #f8fcff 48%,
        #e0f4ff 100%
    );
    color: #164e70;
}

[data-testid="stHeader"] {
    background: rgba(239,250,255,.88);
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #d8f1ff 0%,
        #effaff 100%
    );
    border-right: 1px solid #a5ddf8;
}

h1, h2, h3 {
    color: #075985 !important;
    font-weight: 750 !important;
}

p, label {
    color: #164e70;
}

/* HEADER */

.hero-card {
    display: flex;
    align-items: center;
    gap: 28px;
    width: 100%;
    box-sizing: border-box;

    background: linear-gradient(
        120deg,
        #dff5ff 0%,
        #bceaff 50%,
        #eaf8ff 100%
    );

    border: 1px solid #8bd5f5;
    border-radius: 24px;
    padding: 25px 32px;
    margin: 8px 0 24px;

    box-shadow: 0 12px 30px rgba(2,132,199,.16);

    animation: riseIn .85s ease both;
}

/* ROTATING LOGO */

.logo-ring {
    flex: 0 0 145px;
    width: 145px;
    height: 145px;

    display: flex;
    align-items: center;
    justify-content: center;

    position: relative;
    isolation: isolate;
    border-radius: 50%;

    filter: drop-shadow(
        0 8px 18px rgba(2,132,199,.25)
    );
}

.logo-ring::before {
    content: "";
    position: absolute;
    inset: -7px;
    border-radius: 50%;

    background: conic-gradient(
        from 0deg,
        #ffffff,
        #38bdf8,
        #0284c7,
        #7dd3fc,
        #ffffff
    );

    animation: spinRing 4s linear infinite;
    z-index: -1;
}

.logo-ring::after {
    content: "";
    position: absolute;
    inset: -2px;
    border-radius: 50%;
    background: #eaf8ff;
    z-index: -1;
}

.logo-ring img {
    width: 135px;
    height: 135px;
    object-fit: cover;
    border-radius: 50%;
    border: 4px solid white;
    display: block;
}

.logo-placeholder {
    font-size: 65px;
}

/* HEADER TEXT */

.hero-copy {
    flex: 1;
    min-width: 0;
    text-align: left;
}

.hero-copy h1 {
    color: #075985 !important;
    font-size: clamp(24px, 3vw, 38px);
    font-weight: 850 !important;
    line-height: 1.2;
    letter-spacing: .5px;
    margin: 0 0 10px 0;
}

.hero-copy p {
    color: #23617f !important;
    font-size: 16px;
    font-weight: 500;
    margin: 0;
}

@media (max-width: 640px) {
    .hero-card {
        flex-direction: column;
        text-align: center;
        padding: 22px 15px;
        gap: 18px;
    }

    .hero-copy {
        text-align: center;
    }

    .logo-ring {
        flex-basis: 115px;
        width: 115px;
        height: 115px;
    }

    .logo-ring img {
        width: 105px;
        height: 105px;
    }

    .hero-copy h1 {
        font-size: 25px;
    }
}

/* METRIC CARDS */

.metric-card {
    background: linear-gradient(
        145deg,
        #ffffff,
        #e0f4ff
    );

    padding: 19px 12px;
    border-radius: 17px;
    border: 1px solid #a5ddf8;
    text-align: center;

    box-shadow: 0 5px 17px rgba(14,165,233,.10);

    transition: transform .25s ease,
                box-shadow .25s ease;

    animation: riseIn .7s ease both;
}

.metric-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 24px rgba(2,132,199,.20);
}

.metric-title {
    color: #0369a1;
    font-size: 14px;
    font-weight: 650;
}

.metric-value {
    color: #0284c7;
    font-size: 28px;
    font-weight: 800;
    margin-top: 5px;
}

/* PANELS */

.panel {
    background: rgba(255,255,255,.88);
    border: 1px solid #c5eaff;
    border-radius: 17px;
    padding: 18px 20px;
    margin: 8px 0 17px;
    box-shadow: 0 5px 16px rgba(14,165,233,.07);
    animation: riseIn .65s ease both;
}

/* BUTTONS */

.stButton > button {
    background: linear-gradient(
        90deg,
        #38bdf8,
        #0284c7
    );

    color: white !important;
    border: 0;
    border-radius: 11px;
    font-weight: 700;
    transition: all .25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 18px rgba(2,132,199,.25);
}

/* TABS */

.stTabs [data-baseweb="tab"] {
    color: #0369a1;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: #d8f1ff !important;
    border-radius: 9px;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #f4fbff;
    border: 1px dashed #38bdf8;
    border-radius: 14px;
    padding: 12px;
}

/* FOOTER */

.footer {
    background: linear-gradient(
        90deg,
        #38bdf8,
        #0284c7
    );

    color: white;
    text-align: center;
    padding: 14px;
    border-radius: 13px;
    margin-top: 25px;
}

/* ANIMATIONS */

@keyframes spinRing {
    to {
        transform: rotate(360deg);
    }
}

@keyframes riseIn {
    from {
        opacity: 0;
        transform: translateY(13px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: .01ms !important;
        transition-duration: .01ms !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOGO + DASHBOARD HEADER
# ---------------------------------------------------------

logo_path = Path(__file__).parent / "sentiment_logo.png"

if logo_path.exists():

    logo_b64 = base64.b64encode(
        logo_path.read_bytes()
    ).decode("utf-8")

    logo_html = f"""
        <img src="data:image/png;base64,{logo_b64}"
             alt="Sentiment Analysis Logo">
    """

else:

    logo_html = '<span class="logo-placeholder">🩵</span>'


header_html = f"""
<div class="hero-card">

    <div class="logo-ring">
        {logo_html}
    </div>

    <div class="hero-copy">

        <h1>
            🩵 Sentiment Analysis · Machine Learning Project
        </h1>

        <p>
            Sky Blue Dashboard Edition
        </p>

    </div>

</div>
"""

st.html(header_html)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.markdown("## 🩵 Dashboard Controls")

st.sidebar.markdown(
    "Upload a CSV file with **review** and **sentiment** columns."
)

uploaded_file = st.sidebar.file_uploader(
    "Choose sentiment dataset",
    type=["csv"]
)

st.sidebar.markdown("---")

model_name = st.sidebar.selectbox(
    "Select Machine Learning Algorithm",
    [
        "Multinomial Naive Bayes",
        "Logistic Regression",
        "Random Forest"
    ]
)

st.sidebar.caption("Sky Blue Edition · Animated logo")


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def metric_card(title, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_models():

    return {
        "Multinomial Naive Bayes": MultinomialNB(),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes):

    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_resource(show_spinner=False)
def train_models(reviews, labels):

    X_train, X_test, y_train, y_test = train_test_split(
        list(reviews),
        list(labels),
        test_size=0.20,
        random_state=42,
        stratify=list(labels)
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    trained = {}
    scores = {}

    for name, model in get_models().items():

        model.fit(X_train_vec, y_train)

        pred = model.predict(X_test_vec)

        trained[name] = model

        scores[name] = {
            "Accuracy": accuracy_score(y_test, pred),

            "Precision": precision_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )
        }

    return vectorizer, trained, scores, X_test, y_test


# ---------------------------------------------------------
# EMPTY STATE
# ---------------------------------------------------------

if uploaded_file is None:

    st.markdown("## Welcome to your Sentiment Analysis Studio")

    st.markdown("""
    Upload your dataset from the sidebar to explore sentiment patterns,
    train machine learning algorithms, compare model performance, and
    test new reviews with live predictions.

    **Required CSV columns:** `review` and `sentiment`.
    """)

    st.info(
        "Start by uploading your CSV dataset in the left sidebar."
    )

    st.stop()


# ---------------------------------------------------------
# READ + VALIDATE DATA
# ---------------------------------------------------------

try:

    df = load_dataset(uploaded_file.getvalue())

except Exception as exc:

    st.error(f"Could not read the CSV file: {exc}")
    st.stop()


# Normalize column names

df.columns = df.columns.str.strip().str.lower()

required = {"review", "sentiment"}

if not required.issubset(df.columns):

    st.error(
        "Your CSV must contain these columns: review and sentiment."
    )

    st.stop()


df = df[["review", "sentiment"]].copy()

df["review"] = (
    df["review"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["sentiment"] = (
    df["sentiment"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    (df["review"] != "") &
    (df["sentiment"] != "")
].reset_index(drop=True)


if len(df) < 4:

    st.error("Please provide at least 4 valid review rows.")
    st.stop()


if df["sentiment"].nunique() < 2:

    st.error(
        "The dataset must contain at least two sentiment classes."
    )

    st.stop()


if df["sentiment"].value_counts().min() < 2:

    st.error(
        "Each sentiment class needs at least 2 rows for stratified splitting."
    )

    st.stop()


# ---------------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------------

st.markdown("## Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card("Total Reviews", f"{len(df):,}")

with c2:
    metric_card(
        "Sentiment Classes",
        str(df["sentiment"].nunique())
    )

with c3:
    metric_card(
        "Unique Reviews",
        f"{df['review'].nunique():,}"
    )

with c4:
    metric_card(
        "Empty Values",
        str(int(df.isna().sum().sum()))
    )


# ---------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------

tab_data, tab_analysis, tab_models, tab_predict = st.tabs([
    "📁 Dataset",
    "📊 Analysis",
    "🤖 Model Performance",
    "🔮 Live Prediction"
])


# ---------------------------------------------------------
# DATASET TAB
# ---------------------------------------------------------

with tab_data:

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Sentiment Counts")

    counts = (
        df["sentiment"]
        .value_counts()
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )

    st.dataframe(
        counts,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# ANALYSIS TAB
# ---------------------------------------------------------

with tab_analysis:

    st.markdown("### Sentiment Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.countplot(
        data=df,
        x="sentiment",
        color="#38bdf8",
        ax=ax
    )

    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Sentiment Class Distribution")

    ax.tick_params(
        axis="x",
        rotation=20
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # REVIEW LENGTH DISTRIBUTION

    st.markdown("### Review Length Distribution")

    lengths = df["review"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.histplot(
        lengths,
        bins=25,
        color="#38bdf8",
        ax=ax
    )

    ax.set_xlabel("Words per Review")
    ax.set_ylabel("Frequency")
    ax.set_title("Review Length")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------
# TRAIN MACHINE LEARNING MODELS
# ---------------------------------------------------------

with st.spinner("Training the machine learning models..."):

    try:

        vectorizer, trained_models, results, X_test, y_test = train_models(
            tuple(df["review"].tolist()),
            tuple(df["sentiment"].tolist())
        )

    except Exception as exc:

        st.error(f"Model training failed: {exc}")
        st.stop()


selected_model = trained_models[model_name]

test_vectors = vectorizer.transform(list(X_test))

predictions = selected_model.predict(test_vectors)


# ---------------------------------------------------------
# MODEL PERFORMANCE TAB
# ---------------------------------------------------------

with tab_models:

    st.markdown(f"### Selected Algorithm: {model_name}")

    score_values = {

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

    m1, m2, m3, m4 = st.columns(4)

    for col, (label, value) in zip(
        [m1, m2, m3, m4],
        score_values.items()
    ):

        with col:

            metric_card(
                label,
                f"{value * 100:.2f}%"
            )


    # MODEL COMPARISON

    st.markdown("### Model Comparison")

    comparison = pd.DataFrame(results).T * 100

    st.dataframe(
        comparison.round(2),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    comparison.plot(
        kind="bar",
        ax=ax,
        color=[
            "#38bdf8",
            "#0284c7",
            "#7dd3fc",
            "#075985"
        ]
    )

    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Algorithm")
    ax.set_ylim(0, 105)

    ax.set_title("Machine Learning Model Comparison")

    ax.tick_params(
        axis="x",
        rotation=15
    )

    ax.legend(loc="lower right")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


    # CONFUSION MATRIX

    st.markdown("### Confusion Matrix")

    labels = selected_model.classes_

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1,
        linecolor="white",
        ax=ax
    )

    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("Actual Sentiment")

    ax.set_title(
        f"{model_name} Confusion Matrix"
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


    # CLASSIFICATION REPORT

    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).T.round(3),
        use_container_width=True
    )


# ---------------------------------------------------------
# LIVE PREDICTION TAB
# ---------------------------------------------------------

with tab_predict:

    st.markdown("### Try a New Review")

    review_text = st.text_area(
        "Enter your review",

        placeholder=(
            "Example: The product quality is excellent "
            "and I really love it."
        ),

        height=125
    )

    if st.button(
        "✨ Predict Sentiment",
        use_container_width=True
    ):

        if not review_text.strip():

            st.warning("Please enter a review first.")

        else:

            review_vector = vectorizer.transform([
                review_text.strip()
            ])

            predicted = selected_model.predict(
                review_vector
            )[0]

            st.success(
                f"Predicted Sentiment: {predicted}"
            )

            if hasattr(selected_model, "predict_proba"):

                probs = selected_model.predict_proba(
                    review_vector
                )[0]

                probability_df = pd.DataFrame({

                    "Sentiment": selected_model.classes_,

                    "Confidence (%)": probs * 100

                }).sort_values(
                    "Confidence (%)",
                    ascending=False
                )

                st.markdown("### Prediction Confidence")

                st.dataframe(
                    probability_df.round(2),
                    use_container_width=True,
                    hide_index=True
                )

                fig, ax = plt.subplots(figsize=(8, 3))

                ax.barh(
                    probability_df["Sentiment"],
                    probability_df["Confidence (%)"],
                    color="#38bdf8"
                )

                ax.set_xlim(0, 100)
                ax.set_xlabel("Confidence (%)")
                ax.set_title("Prediction Confidence")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.html("""
<div class="footer">

    <b>🩵 Sentiment Analysis · Machine Learning Project</b>

    <br>

    Sky Blue Dashboard Edition

</div>
""")
