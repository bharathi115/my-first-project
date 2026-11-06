
import base64
import io
import random
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


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sentiment Analysis | Machine Learning",
    page_icon="🩵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. SKY BLUE PROFESSIONAL THEME
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
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

[data-testid="stSidebar"] * {
    color: #075985;
}

h1, h2, h3 {
    color: #075985 !important;
    font-weight: 750 !important;
}

p, label {
    color: #164e70;
}

.hero-card {
    display: flex;
    align-items: center;
    gap: 28px;
    background: linear-gradient(
        120deg,
        #dff5ff,
        #bceaff,
        #eaf8ff
    );
    border: 1px solid #8bd5f5;
    border-radius: 24px;
    padding: 25px 32px;
    margin: 8px 0 24px;
    box-shadow: 0 12px 30px rgba(2,132,199,.16);
    animation: riseIn .7s ease both;
}

.logo-ring {
    flex: 0 0 125px;
    width: 125px;
    height: 125px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    isolation: isolate;
    border-radius: 50%;
}

.logo-ring::before {
    content: "";
    position: absolute;
    inset: -6px;
    border-radius: 50%;
    background: conic-gradient(
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
    inset: -1px;
    border-radius: 50%;
    background: #eaf8ff;
    z-index: -1;
}

.logo-ring img {
    width: 115px;
    height: 115px;
    object-fit: cover;
    border-radius: 50%;
    border: 4px solid white;
}

.logo-placeholder {
    font-size: 60px;
}

.hero-copy {
    flex: 1;
    min-width: 0;
}

.hero-copy h1 {
    color: #075985 !important;
    font-size: clamp(23px, 3vw, 36px);
    font-weight: 850 !important;
    margin: 0 0 10px;
}

.hero-copy p {
    color: #23617f !important;
    font-size: 15px;
    margin: 0;
}

.metric-card {
    background: linear-gradient(145deg, #ffffff, #e0f4ff);
    padding: 20px 12px;
    border-radius: 17px;
    border: 1px solid #a5ddf8;
    text-align: center;
    box-shadow: 0 5px 17px rgba(14,165,233,.10);
    transition: transform .25s ease;
    min-height: 105px;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-title {
    color: #0369a1;
    font-size: 13px;
    font-weight: 650;
}

.metric-value {
    color: #0284c7;
    font-size: 27px;
    font-weight: 800;
    margin-top: 7px;
}

.panel {
    background: rgba(255,255,255,.92);
    border: 1px solid #c5eaff;
    border-radius: 17px;
    padding: 20px;
    margin: 8px 0 17px;
    box-shadow: 0 5px 16px rgba(14,165,233,.07);
}

.section-heading {
    font-size: 21px;
    color: #075985;
    font-weight: 800;
    margin: 15px 0;
}

.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #0284c7);
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

.stTabs [data-baseweb="tab"] {
    color: #0369a1;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: #d8f1ff !important;
    border-radius: 9px;
}

[data-testid="stFileUploader"] {
    background: #f4fbff;
    border: 1px dashed #38bdf8;
    border-radius: 14px;
    padding: 12px;
}

.footer {
    background: linear-gradient(90deg, #38bdf8, #0284c7);
    color: white;
    text-align: center;
    padding: 15px;
    border-radius: 13px;
    margin-top: 25px;
}

@keyframes spinRing {
    to { transform: rotate(360deg); }
}

@keyframes riseIn {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 640px) {
    .hero-card {
        flex-direction: column;
        text-align: center;
        padding: 20px;
    }
    .hero-copy {
        text-align: center;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        transition-duration: .01ms !important;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DASHBOARD HEADER AND LOGO
# =========================================================

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


st.html(f"""
<div class="hero-card">

    <div class="logo-ring">
        {logo_html}
    </div>

    <div class="hero-copy">
        <h1>🩵 Sentiment Analysis Dashboard</h1>
        <p>Natural Language Processing · Machine Learning · Data Analytics</p>
        <p>Sky Blue Professional Dashboard Edition</p>
    </div>

</div>
""")


# =========================================================
# 4. SIDEBAR
# =========================================================

st.sidebar.markdown("# 🩵 SENTIMENT ANALYSIS")
st.sidebar.caption("INTELLIGENT ANALYTICS PLATFORM")

st.sidebar.divider()

st.sidebar.markdown("### DATASET")

uploaded_file = st.sidebar.file_uploader(
    "Upload Sentiment Dataset",
    type=["csv"]
)

st.sidebar.divider()

st.sidebar.markdown("### MODEL CONFIGURATION")

model_name = st.sidebar.selectbox(
    "Select Machine Learning Algorithm",
    [
        "Multinomial Naive Bayes",
        "Logistic Regression",
        "Random Forest"
    ]
)

test_percentage = st.sidebar.select_slider(
    "Testing Data Percentage",
    options=[10, 15, 20, 25, 30, 35, 40],
    value=20
)

test_size = test_percentage / 100

st.sidebar.divider()

st.sidebar.caption("Python | NLP | TF-IDF | Machine Learning")
st.sidebar.caption("Sky Blue Edition")


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def metric_card(title, value):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section_title(title):

    st.markdown(
        f'<div class="section-heading">{title}</div>',
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


# =========================================================
# 6. DATASET GENERATOR
# =========================================================

def generate_sample_dataset():

    random.seed(42)

    positive_reviews = [
        "This product is excellent and I really love it",
        "Amazing quality and very good service",
        "I am extremely happy with this purchase",
        "The product works perfectly and looks great",
        "Excellent experience and fast delivery",
        "Very useful product and worth the money",
        "The quality is outstanding and impressive",
        "I would definitely recommend this product",
        "Fantastic experience and great customer support",
        "Really satisfied with the product",
        "Good quality and reasonable price",
        "I am happy with the service",
        "Wonderful product with excellent performance",
        "The delivery was quick and the product is perfect",
        "Great value for money and highly recommended",
        "This is one of my best purchases",
        "Very comfortable and easy to use",
        "Excellent product with beautiful design",
        "I absolutely love this product",
        "The experience was smooth and satisfying"
    ]

    negative_reviews = [
        "This product is terrible and very disappointing",
        "Poor quality and bad customer service",
        "I am very unhappy with this purchase",
        "The product stopped working after one day",
        "Very slow delivery and damaged packaging",
        "Not worth the money at all",
        "The quality is extremely poor",
        "I would never recommend this product",
        "Terrible experience and no customer support",
        "I am completely dissatisfied",
        "Bad quality and expensive price",
        "The service was very disappointing",
        "Worst product I have ever purchased",
        "The delivery was late and the product was damaged",
        "Waste of money and poor performance",
        "I regret buying this product",
        "Very uncomfortable and difficult to use",
        "Poor design and bad quality",
        "I absolutely hate this product",
        "The experience was frustrating and disappointing"
    ]

    rows = []

    for _ in range(500):

        rows.append({
            "review": random.choice(positive_reviews),
            "sentiment": "positive"
        })

        rows.append({
            "review": random.choice(negative_reviews),
            "sentiment": "negative"
        })

    return pd.DataFrame(rows).sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)


# =========================================================
# 7. DATA VALIDATION FUNCTION
# =========================================================

def validate_dataset(raw_df, clean_df):

    checks = []

    original_columns = [
        str(col).strip().lower()
        for col in raw_df.columns
    ]

    required_columns = {"review", "sentiment"}

    checks.append({
        "Validation": "Required columns present",
        "Status": (
            "PASS"
            if required_columns.issubset(set(original_columns))
            else "FAIL"
        ),
        "Details": "review and sentiment columns"
    })

    missing_values = int(
        raw_df[["review", "sentiment"]].isna().sum().sum()
    ) if required_columns.issubset(set(original_columns)) else None

    checks.append({
        "Validation": "Missing values",
        "Status": "PASS" if missing_values == 0 else "FAIL",
        "Details": (
            f"{missing_values} missing values"
            if missing_values is not None
            else "Required columns unavailable"
        )
    })

    if required_columns.issubset(set(original_columns)):

        temp = raw_df.copy()
        temp.columns = original_columns

        blank_reviews = int(
            temp["review"].fillna("").astype(str).str.strip().eq("").sum()
        )

        blank_labels = int(
            temp["sentiment"].fillna("").astype(str).str.strip().eq("").sum()
        )

        duplicate_count = int(
            temp.duplicated(subset=["review", "sentiment"]).sum()
        )

        valid_labels = (
            temp["sentiment"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        invalid_labels = int(
            (~valid_labels.isin(["positive", "negative"])).sum()
        )

    else:

        blank_reviews = 0
        blank_labels = 0
        duplicate_count = 0
        invalid_labels = 0

    checks.append({
        "Validation": "Empty reviews",
        "Status": "PASS" if blank_reviews == 0 else "FAIL",
        "Details": f"{blank_reviews} empty reviews"
    })

    checks.append({
        "Validation": "Empty sentiment labels",
        "Status": "PASS" if blank_labels == 0 else "FAIL",
        "Details": f"{blank_labels} empty labels"
    })

    checks.append({
        "Validation": "Duplicate records",
        "Status": "PASS" if duplicate_count == 0 else "WARNING",
        "Details": f"{duplicate_count} duplicate records"
    })

    checks.append({
        "Validation": "Valid sentiment labels",
        "Status": "PASS" if invalid_labels == 0 else "FAIL",
        "Details": f"{invalid_labels} invalid labels"
    })

    checks.append({
        "Validation": "Minimum dataset size",
        "Status": "PASS" if len(clean_df) >= 4 else "FAIL",
        "Details": f"{len(clean_df)} usable records"
    })

    class_count = clean_df["sentiment"].nunique()

    checks.append({
        "Validation": "Positive and negative classes",
        "Status": "PASS" if class_count == 2 else "FAIL",
        "Details": f"{class_count} sentiment classes found"
    })

    smallest_class = (
        int(clean_df["sentiment"].value_counts().min())
        if not clean_df.empty else 0
    )

    checks.append({
        "Validation": "Minimum class records",
        "Status": "PASS" if smallest_class >= 2 else "FAIL",
        "Details": f"Smallest class contains {smallest_class} records"
    })

    checks.append({
        "Validation": "Review text quality",
        "Status": (
            "PASS"
            if not clean_df.empty and clean_df["review"].str.len().min() >= 2
            else "WARNING"
        ),
        "Details": "Checks that usable review text is present"
    })

    return pd.DataFrame(checks)


# =========================================================
# 8. TRAIN MACHINE LEARNING MODELS
# =========================================================

@st.cache_resource(show_spinner=False)
def train_models(reviews, labels, test_size):

    X_train, X_test, y_train, y_test = train_test_split(
        list(reviews),
        list(labels),
        test_size=test_size,
        random_state=42,
        stratify=list(labels)
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english"
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

    return (
        vectorizer,
        trained,
        scores,
        X_test,
        y_test
    )


# =========================================================
# 9. LOAD DATASET
# =========================================================

if uploaded_file is not None:

    try:
        raw_df = load_dataset(uploaded_file.getvalue())

    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

else:

    raw_df = generate_sample_dataset()

    st.info(
        "Sample dataset is active. Upload your own CSV "
        "from the sidebar to analyze your data."
    )


# =========================================================
# 10. CHECK REQUIRED COLUMNS
# =========================================================

raw_df.columns = raw_df.columns.astype(str).str.strip().str.lower()

if not {"review", "sentiment"}.issubset(raw_df.columns):

    st.error(
        "Your CSV must contain exactly these required fields: "
        "review and sentiment."
    )

    st.stop()


# =========================================================
# 11. CLEAN DATASET
# =========================================================

df = raw_df[["review", "sentiment"]].copy()

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
    .str.lower()
)

df = df[
    (df["review"] != "") &
    (df["sentiment"].isin(["positive", "negative"]))
]

df = df.drop_duplicates().reset_index(drop=True)


# =========================================================
# 12. VALIDATION RESULTS
# =========================================================

validation_df = validate_dataset(raw_df, df)

passed = int(
    (validation_df["Status"] == "PASS").sum()
)

failed = int(
    (validation_df["Status"] == "FAIL").sum()
)

warnings = int(
    (validation_df["Status"] == "WARNING").sum()
)

validation_percentage = (
    passed / len(validation_df) * 100
)


# =========================================================
# 13. DASHBOARD METRICS
# =========================================================

section_title("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card("TOTAL REVIEWS", f"{len(df):,}")

with c2:
    metric_card(
        "POSITIVE REVIEWS",
        f"{int((df['sentiment'] == 'positive').sum()):,}"
    )

with c3:
    metric_card(
        "NEGATIVE REVIEWS",
        f"{int((df['sentiment'] == 'negative').sum()):,}"
    )

with c4:
    metric_card(
        "VALIDATION SCORE",
        f"{validation_percentage:.0f}%"
    )


# =========================================================
# 14. DASHBOARD NAVIGATION
# =========================================================

tab_data, tab_validation, tab_analysis, tab_models, tab_predict = st.tabs([
    "📁 Dataset",
    "✅ Validation",
    "📊 Analysis & Graphs",
    "🤖 Model Performance",
    "🔮 Live Prediction"
])


# =========================================================
# 15. DATASET TAB
# =========================================================

with tab_data:

    section_title("Dataset Management")

    search_text = st.text_input(
        "🔍 Search Review",
        placeholder="Search customer reviews..."
    )

    filter_sentiment = st.selectbox(
        "Filter Sentiment",
        ["All", "Positive", "Negative"]
    )

    display_df = df.copy()

    if search_text:

        display_df = display_df[
            display_df["review"].str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

    if filter_sentiment == "Positive":

        display_df = display_df[
            display_df["sentiment"] == "positive"
        ]

    elif filter_sentiment == "Negative":

        display_df = display_df[
            display_df["sentiment"] == "negative"
        ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(f"Showing {len(display_df)} records")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Clean Dataset",
        data=csv_data,
        file_name="sentiment_dataset.csv",
        mime="text/csv"
    )


# =========================================================
# 16. VALIDATION TAB
# =========================================================

with tab_validation:

    section_title("Dataset Validation Report")

    st.markdown("""
    Check the quality and structure of your uploaded dataset
    before machine learning model training.
    """)

    v1, v2, v3, v4 = st.columns(4)

    with v1:
        metric_card("TOTAL CHECKS", str(len(validation_df)))

    with v2:
        metric_card("PASSED", str(passed))

    with v3:
        metric_card("FAILED", str(failed))

    with v4:
        metric_card("WARNINGS", str(warnings))

    st.write("")

    st.markdown("### Overall Validation Score")

    st.progress(validation_percentage / 100)

    st.write(f"Validation score: **{validation_percentage:.1f}%**")

    st.markdown("### Individual Validation Results")

    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True
    )

    if failed == 0:

        st.success("All critical validation checks passed.")

    else:

        st.error(
            f"{failed} validation checks failed. "
            "Please review the details above."
        )

    if warnings > 0:

        st.warning(
            "Some data quality warnings were found. "
            "Review duplicate records or other warnings."
        )

    st.markdown("### Data Quality Summary")

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:

        st.markdown("#### Sentiment Class Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))

        df["sentiment"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Records")
        ax.set_title("Class Balance")
        ax.tick_params(axis="x", rotation=0)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with quality_col2:

        st.markdown("#### Validation Status")

        status_counts = validation_df["Status"].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Validation Check Results")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# =========================================================
# 17. ANALYSIS AND GRAPHS TAB
# =========================================================

with tab_analysis:

    section_title("Exploratory Data Analysis")

    analysis_df = df.copy()

    analysis_df["word_count"] = (
        analysis_df["review"].str.split().str.len()
    )

    analysis_df["review_length"] = (
        analysis_df["review"].str.len()
    )

    # GRAPH 1: SENTIMENT COUNT

    st.markdown("### 1. Sentiment Distribution")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.countplot(
            data=analysis_df,
            x="sentiment",
            hue="sentiment",
            palette=["#38bdf8", "#0284c7"],
            legend=False,
            ax=ax
        )

        ax.set_title("Positive vs Negative Reviews")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Reviews")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # GRAPH 2: SENTIMENT PIE CHART

    with col2:

        sentiment_counts = analysis_df["sentiment"].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Sentiment Percentage")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # GRAPH 3: REVIEW LENGTH

    st.markdown("### 2. Review Length Analysis")

    col3, col4 = st.columns(2)

    with col3:

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            data=analysis_df,
            x="review_length",
            hue="sentiment",
            bins=25,
            kde=True,
            ax=ax
        )

        ax.set_title("Character Length Distribution")
        ax.set_xlabel("Characters")
        ax.set_ylabel("Frequency")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col4:

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.boxplot(
            data=analysis_df,
            x="sentiment",
            y="word_count",
            hue="sentiment",
            legend=False,
            ax=ax
        )

        ax.set_title("Word Count by Sentiment")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Words")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # GRAPH 4: AVERAGE REVIEW LENGTH

    st.markdown("### 3. Average Review Length")

    avg_length = (
        analysis_df.groupby("sentiment")["review_length"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.barplot(
        data=avg_length,
        x="sentiment",
        y="review_length",
        hue="sentiment",
        legend=False,
        ax=ax
    )

    ax.set_title("Average Review Length by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Average Characters")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # GRAPH 5: AVERAGE WORD COUNT

    st.markdown("### 4. Average Word Count")

    avg_words = (
        analysis_df.groupby("sentiment")["word_count"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    sns.barplot(
        data=avg_words,
        x="sentiment",
        y="word_count",
        hue="sentiment",
        legend=False,
        ax=ax
    )

    ax.set_title("Average Words per Review")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Average Word Count")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # STATISTICS

    st.markdown("### 5. Statistical Summary")

    st.dataframe(
        analysis_df[
            ["review_length", "word_count"]
        ].describe().round(2),
        use_container_width=True
    )


# =========================================================
# 18. TRAIN MODELS
# =========================================================

if len(df) < 4 or df["sentiment"].nunique() < 2:

    st.warning(
        "Not enough valid data to train models. "
        "Please upload a larger dataset with both sentiment classes."
    )

    st.stop()

if df["sentiment"].value_counts().min() < 2:

    st.warning(
        "Each sentiment class needs at least two records."
    )

    st.stop()

try:

    with st.spinner("Training all machine learning models..."):

        (
            vectorizer,
            trained_models,
            results,
            X_test,
            y_test
        ) = train_models(
            tuple(df["review"].tolist()),
            tuple(df["sentiment"].tolist()),
            test_size
        )

except Exception as exc:

    st.error(f"Model training failed: {exc}")
    st.stop()


selected_model = trained_models[model_name]

test_vectors = vectorizer.transform(list(X_test))

predictions = selected_model.predict(test_vectors)


# =========================================================
# 19. MODEL PERFORMANCE TAB
# =========================================================

with tab_models:

    section_title("Machine Learning Model Evaluation")

    st.write(f"Selected Algorithm: **{model_name}**")

    score_values = {
        "Accuracy": accuracy_score(y_test, predictions),

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
            metric_card(label, f"{value * 100:.2f}%")

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
    ax.set_xlabel("Machine Learning Algorithm")
    ax.set_ylim(0, 105)
    ax.set_title("Machine Learning Model Comparison")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="lower right")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
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
    ax.set_title(f"{model_name} Confusion Matrix")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # CLASSIFICATION REPORT

    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # ACTUAL VS PREDICTED

    st.markdown("### Actual vs Predicted Results")

    result_df = pd.DataFrame({
        "Review": list(X_test),
        "Actual Sentiment": list(y_test),
        "Predicted Sentiment": predictions
    })

    st.dataframe(
        result_df.head(50),
        use_container_width=True,
        hide_index=True
    )

    result_csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Prediction Results",
        data=result_csv,
        file_name="sentiment_prediction_results.csv",
        mime="text/csv"
    )


# =========================================================
# 20. LIVE PREDICTION TAB
# =========================================================

with tab_predict:

    section_title("Live Sentiment Prediction")

    st.markdown("""
    Enter a new customer review. The selected machine learning
    algorithm will predict its sentiment.
    """)

    review_text = st.text_area(
        "Enter Customer Review",
        placeholder=(
            "Example: The product quality is excellent "
            "and I really love it."
        ),
        height=140
    )

    if st.button(
        "✨ Predict Sentiment",
        use_container_width=True
    ):

        if not review_text.strip():

            st.warning("Please enter a review.")

        else:

            review_vector = vectorizer.transform([
                review_text.strip()
            ])

            predicted = selected_model.predict(
                review_vector
            )[0]

            st.markdown("### Prediction Result")

            if predicted == "positive":
                st.success("😊 POSITIVE SENTIMENT")
            else:
                st.error("😞 NEGATIVE SENTIMENT")

            if hasattr(selected_model, "predict_proba"):

                probabilities = selected_model.predict_proba(
                    review_vector
                )[0]

                probability_df = pd.DataFrame({
                    "Sentiment": selected_model.classes_,
                    "Confidence (%)": probabilities * 100
                }).sort_values(
                    "Confidence (%)",
                    ascending=False
                )

                predicted_confidence = probability_df.loc[
                    probability_df["Sentiment"] == predicted,
                    "Confidence (%)"
                ].iloc[0]

                metric_card(
                    "PREDICTION CONFIDENCE",
                    f"{predicted_confidence:.2f}%"
                )

                st.markdown("### Sentiment Probability")

                fig, ax = plt.subplots(figsize=(8, 4))

                ax.barh(
                    probability_df["Sentiment"],
                    probability_df["Confidence (%)"]
                )

                ax.set_xlim(0, 100)
                ax.set_xlabel("Confidence (%)")
                ax.set_title("Prediction Probability")

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.dataframe(
                    probability_df.round(2),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("### Submitted Review")

            st.info(review_text)


# =========================================================
# 21. FOOTER
# =========================================================

st.html("""
<div class="footer">
    <b>🩵 Sentiment Analysis · Machine Learning Project</b>
    <br>
    Sky Blue Dashboard Edition
    <br>
    Python | NLP | TF-IDF | Machine Learning
</div>
""")
