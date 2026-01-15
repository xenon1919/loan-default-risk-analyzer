import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap

# -------------------------
# App Config
# -------------------------
st.set_page_config(
    page_title="Loan Default Risk Analyzer",
    layout="wide"
)

st.title("🏦 Loan Default Risk Analyzer")
st.caption("From statistics to decisions. No vibes involved.")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("../data/raw/loan_default_dataset_v2.csv")

df = load_data()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Dataset Overview",
        "EDA",
        "Model Performance",
        "Predict Default Risk",
        "Explain Prediction (SHAP)"
    ]
)


# -------------------------
# Feature Engineering
# -------------------------
df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
df["income_per_year_employed"] = df["annual_income"] / (df["employment_years"] + 1)

X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
# SHAP explainer (logistic regression friendly)
explainer = shap.LinearExplainer(
    model,
    X_train_scaled,
    feature_names=X.columns
)

# -------------------------
# Section 1: Dataset Overview
# -------------------------
if section == "Dataset Overview":
    st.subheader("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])
        st.metric("Features", df.shape[1] - 1)

    with col2:
        default_rate = df["default"].mean()
        st.metric("Default Rate", f"{default_rate:.2%}")

    st.dataframe(df.head())

# -------------------------
# Section 2: EDA
# -------------------------
elif section == "EDA":
    st.subheader("🔍 Exploratory Data Analysis")

    feature = st.selectbox(
        "Select a feature",
        ["annual_income", "loan_amount", "credit_score", "interest_rate"]
    )

    fig, ax = plt.subplots()
    ax.hist(df[feature], bins=30)
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)

    st.markdown(
        f"""
        **Why this matters:**  
        The distribution of `{feature}` affects probability estimates, 
        scaling behavior, and model stability.
        """
    )

# -------------------------
# Section 3: Model Performance
# -------------------------
elif section == "Model Performance":
    st.subheader("📈 Model Performance")

    preds = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, preds)

    st.metric("ROC-AUC Score", f"{auc:.3f}")

    st.markdown(
        """
        **Interpretation:**
        - 0.5 → random guessing  
        - 0.7+ → usable  
        - 0.8+ → strong  

        This model is doing real work, not guessing politely.
        """
    )

# -------------------------
# Section 4: Prediction
# -------------------------
elif section == "Predict Default Risk":
    st.subheader("🧮 Predict Default Risk")

    age = st.slider("Age", 21, 65, 30)
    income = st.number_input("Annual Income", 150000, 2000000, 600000)
    credit_score = st.slider("Credit Score", 300, 850, 680)
    loan_amount = st.number_input("Loan Amount", 50000, 3000000, 800000)
    interest_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
    employment_years = st.slider("Employment Years", 0, 35, 5)

    loan_to_income = loan_amount / income
    income_per_year_employed = income / (employment_years + 1)

    input_data = np.array([[
        age,
        income,
        credit_score,
        loan_amount,
        interest_rate,
        employment_years,
        loan_to_income,
        income_per_year_employed
    ]])



elif section == "Explain Prediction (SHAP)":
    st.subheader("🧠 Explain Prediction with SHAP")

    st.markdown(
        """
        This section explains **why** the model predicts default risk.
        Positive values increase risk.
        Negative values decrease risk.
        """
    )

    # Reuse same inputs
    age = st.slider("Age", 21, 65, 30)
    income = st.number_input("Annual Income", 150000, 2000000, 600000)
    credit_score = st.slider("Credit Score", 300, 850, 680)
    loan_amount = st.number_input("Loan Amount", 50000, 3000000, 800000)
    interest_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
    employment_years = st.slider("Employment Years", 0, 35, 5)

    loan_to_income = loan_amount / income
    income_per_year_employed = income / (employment_years + 1)

    input_data = pd.DataFrame([{
        "age": age,
        "annual_income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "employment_years": employment_years,
        "loan_to_income": loan_to_income,
        "income_per_year_employed": income_per_year_employed
    }])

    input_scaled = scaler.transform(input_data)

    shap_values = explainer(input_scaled)

    st.markdown("### 🔎 Feature Contributions")

    shap_df = pd.DataFrame({
        "feature": X.columns,
        "shap_value": shap_values.values[0]
    }).sort_values(by="shap_value", ascending=False)

    st.dataframe(shap_df)

    st.markdown(
        """
        **How to read this:**
        - Positive SHAP value → increases default risk
        - Negative SHAP value → reduces default risk
        - Larger magnitude → stronger influence
        """
    )


    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]

    st.metric("Predicted Default Probability", f"{prob:.2%}")

    if prob > 0.5:
        st.error("⚠️ High default risk")
    else:
        st.success("✅ Low default risk")
