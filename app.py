import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "dataset" / "creditcard.csv"


# =========================================================
# FILE CHECK
# =========================================================

if not MODEL_PATH.exists():
    st.error(f"❌ Model file not found:\n{MODEL_PATH}")
    st.stop()

if not SCALER_PATH.exists():
    st.error(f"❌ Scaler file not found:\n{SCALER_PATH}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"❌ Dataset not found:\n{DATA_PATH}")
    st.stop()


# =========================================================
# LOAD MODEL / SCALER / DATA
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


model = load_model()
scaler = load_scaler()
df = load_dataset()


# =========================================================
# SESSION STATE
# =========================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🛡️ FraudShield AI")

    st.caption("Credit Card Fraud Detection")

    st.divider()

    st.markdown("### 🎨 Appearance")

    dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode
    )

    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()

    st.divider()

    st.markdown("### ⚙️ System Status")

    st.success("✓ Model loaded")
    st.success("✓ Scaler loaded")
    st.success("✓ Dataset loaded")

    st.divider()

    st.markdown("### 🤖 Model")

    st.caption("Random Forest Classifier")
    st.caption("30 input features")
    st.caption(f"{len(df):,} transactions")


# =========================================================
# THEME
# =========================================================

if st.session_state.dark_mode:

    background = "#0b1220"
    secondary_bg = "#111827"
    card_bg = "#172033"
    text = "#f8fafc"
    muted = "#94a3b8"
    border = "#263247"
    input_bg = "#111827"

else:

    background = "#f5f7fb"
    secondary_bg = "#ffffff"
    card_bg = "#ffffff"
    text = "#172033"
    muted = "#64748b"
    border = "#d9e0ea"
    input_bg = "#ffffff"


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
    f"""
    <style>

    /* ==============================
       GLOBAL
       ============================== */

    .stApp {{
        background: {background};
        color: {text};
    }}

    .main {{
        background: {background};
    }}

    .block-container {{
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {{
        background: {secondary_bg};
        border-right: 1px solid {border};
    }}

    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}

    /* ==============================
       HEADINGS
       ============================== */

    h1, h2, h3, h4, h5, h6 {{
        color: {text} !important;
    }}

    p, label, span {{
        color: {text};
    }}

    .stCaption {{
        color: {muted} !important;
    }}

    /* ==============================
       METRIC CARDS
       ============================== */

    div[data-testid="stMetric"] {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {muted} !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {text} !important;
    }}

    /* ==============================
       INPUTS
       ============================== */

    div[data-baseweb="input"] {{
        background: {input_bg};
        border-radius: 10px;
    }}

    div[data-baseweb="input"] input {{
        color: {text} !important;
        background: {input_bg} !important;
    }}

    div[data-baseweb="select"] > div {{
        background: {input_bg};
    }}

    /* ==============================
       BUTTON
       ============================== */

    .stButton > button {{
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
    }}

    /* ==============================
       EXPANDER
       ============================== */

    details {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 14px;
    }}

    /* ==============================
       DATAFRAME
       ============================== */

    div[data-testid="stDataFrame"] {{
        border: 1px solid {border};
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ==============================
       DIVIDER
       ============================== */

    hr {{
        border-color: {border};
    }}

    /* ==============================
       INFO BOX
       ============================== */

    div[data-testid="stAlert"] {{
        border-radius: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.title("🛡️ FraudShield AI")

st.caption(
    "AI-powered credit card transaction risk detection"
)

st.write(
    "Analyze a transaction and estimate its potential fraud risk."
)


# =========================================================
# DATASET STATISTICS
# =========================================================

total_transactions = len(df)

normal_transactions = int(
    (df["Class"] == 0).sum()
)

fraud_transactions = int(
    (df["Class"] == 1).sum()
)

fraud_percentage = (
    fraud_transactions / total_transactions
) * 100


st.markdown("### 📊 Dataset Overview")


m1, m2, m3, m4 = st.columns(4)


with m1:
    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with m2:
    st.metric(
        "Normal Transactions",
        f"{normal_transactions:,}"
    )

with m3:
    st.metric(
        "Fraud Cases",
        f"{fraud_transactions:,}"
    )

with m4:
    st.metric(
        "Fraud Rate",
        f"{fraud_percentage:.2f}%"
    )


st.divider()


# =========================================================
# TRANSACTION ANALYZER
# =========================================================

st.header("🔍 Transaction Analyzer")

st.caption(
    "Enter the basic transaction details. "
    "The technical V1–V28 features are automatically handled "
    "by the system."
)

# =========================================================
# SAMPLE TRANSACTIONS
# =========================================================

st.markdown("#### ⚡ Quick Demo")

sample_col1, sample_col2 = st.columns(2)

with sample_col1:
    normal_sample = st.button(
        "✅ Load Normal Sample",
        width="stretch"
    )

with sample_col2:
    fraud_sample = st.button(
        "🚨 Load Fraud Sample",
        width="stretch"
    )


# Default values
default_time = 10000.0
default_amount = 100.0


# Sample values
if normal_sample:
    default_time = 50000.0
    default_amount = 120.0

if fraud_sample:
    default_time = 40000.0
    default_amount = 1500.0

input_col1, input_col2 = st.columns(2)


with input_col1:

    st.markdown("#### 🕒 Transaction Time")

    time_value = st.number_input(
        "Transaction time (seconds)",
        min_value=0.0,
        max_value=172800.0,
        value=default_time,
        step=100.0,
        help="Time value from the original transaction dataset."
    )


with input_col2:

    st.markdown("#### 💳 Transaction Amount")

    amount_value = st.number_input(
        "Transaction amount",
        min_value=0.0,
        max_value=30000.0,
        value=default_amount,
        step=10.0,
        help="Transaction amount."
    )


st.write("")


analyze = st.button(
    "🔎 Analyze Transaction",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if analyze:

    try:

        feature_names = [
            "Time",
            *[f"V{i}" for i in range(1, 29)],
            "Amount"
        ]

        # -------------------------------------------------
        # Hidden technical features
        # -------------------------------------------------

        technical_features = np.zeros(28)

        values = [
            time_value,
            *technical_features,
            amount_value
        ]

        input_data = pd.DataFrame(
            [values],
            columns=feature_names
        )

        # -------------------------------------------------
        # SCALE
        # -------------------------------------------------

        input_scaled = scaler.transform(
            input_data
        )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        probability_percent = probability * 100

        # -------------------------------------------------
        # RISK LEVEL
        # -------------------------------------------------

        if probability_percent >= 70:
            risk_level = "HIGH"

        elif probability_percent >= 30:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.divider()

        st.header("📋 Analysis Result")


        if prediction == 1:

            st.error(
                "🚨 Potential Fraudulent Transaction"
            )

            st.warning(
                "The model detected a potentially suspicious "
                "transaction pattern."
            )

        else:

            st.success(
                "✅ Transaction Appears Normal"
            )

            st.info(
                "The model did not detect a strong fraud signal."
            )


        # -------------------------------------------------
        # RESULT METRICS
        # -------------------------------------------------

        r1, r2, r3 = st.columns(3)


        with r1:

            st.metric(
                "Fraud Probability",
                f"{probability_percent:.2f}%"
            )


        with r2:

            st.metric(
                "Risk Level",
                risk_level
            )


        with r3:

            st.metric(
                "Transaction Amount",
                f"₹{amount_value:,.2f}"
            )


        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        st.markdown("#### Fraud Probability")

        st.progress(
            float(
                min(
                    max(probability, 0.0),
                    1.0
                )
            )
        )

        st.caption(
            f"Estimated fraud probability: "
            f"{probability_percent:.2f}%"
        )


        # -------------------------------------------------
        # TRANSACTION SUMMARY
        # -------------------------------------------------

        st.markdown("#### 🧾 Transaction Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write(
                f"**Transaction Time:** "
                f"{time_value:,.0f} seconds"
            )

        with summary_col2:

            st.write(
                f"**Transaction Amount:** "
                f"₹{amount_value:,.2f}"
            )


    except Exception as error:

        st.error("❌ Prediction failed.")

        st.code(
            str(error)
        )


# =========================================================
# ANALYTICS
# =========================================================

st.divider()

st.header("📊 Fraud Analytics")


chart1, chart2 = st.columns(2)


# =========================================================
# NORMAL VS FRAUD
# =========================================================

with chart1:

    st.subheader("Normal vs Fraud")

    distribution = pd.DataFrame(
        {
            "Transactions": [
                normal_transactions,
                fraud_transactions
            ]
        },
        index=[
            "Normal",
            "Fraud"
        ]
    )

    st.bar_chart(
        distribution
    )


# =========================================================
# TRANSACTION PERCENTAGE
# =========================================================

with chart2:

    st.subheader("Transaction Percentage")

    percentage_data = pd.DataFrame(
        {
            "Percentage": [
                (
                    normal_transactions /
                    total_transactions
                ) * 100,

                (
                    fraud_transactions /
                    total_transactions
                ) * 100
            ]
        },
        index=[
            "Normal",
            "Fraud"
        ]
    )

    st.bar_chart(
        percentage_data
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.header("🤖 Model Performance")


@st.cache_data
def evaluate_model(data):

    X = data.drop(
        "Class",
        axis=1
    )

    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    y_pred = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        cm
    )


with st.expander(
    "📈 View Model Evaluation"
):

    with st.spinner(
        "Evaluating Random Forest model..."
    ):

        (
            accuracy,
            precision,
            recall,
            f1,
            cm
        ) = evaluate_model(df)


    p1, p2, p3, p4 = st.columns(4)


    with p1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with p2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )


    with p3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )


    with p4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )


    st.markdown("#### Confusion Matrix")


    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Normal",
            "Actual Fraud"
        ],
        columns=[
            "Predicted Normal",
            "Predicted Fraud"
        ]
    )


    st.dataframe(
        cm_df,
        use_container_width=True
    )


# =========================================================
# ABOUT
# =========================================================

st.divider()

with st.expander(
    "ℹ️ About FraudShield AI"
):

    st.write(
        """
        FraudShield AI is a machine-learning based
        credit-card fraud detection system.

        The Random Forest model uses 30 transaction
        features:

        • Time
        • V1–V28
        • Amount

        V1–V28 are anonymized numerical features from
        the original dataset. They are hidden from the
        normal user interface to keep the application
        simple and user-friendly.

        The dashboard provides:

        • Transaction risk analysis
        • Fraud probability
        • Risk classification
        • Dataset statistics
        • Fraud analytics
        • Model performance
        • Confusion matrix
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🛡️ FraudShield AI • Random Forest • Streamlit"
)