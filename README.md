
# Loan Default Risk Analyzer 🚀

An end-to-end **Machine Learning + Explainable AI** project that predicts loan default risk and explains *why* a model makes each decision.

This repository demonstrates how statistics, probability, linear algebra, and gradients come together in a real-world ML application — wrapped inside an interactive **Streamlit app with SHAP explanations**.

---

## 🔍 Project Highlights

- 📊 Structured EDA using statistics & probability
- 🧠 Logistic Regression with solid mathematical intuition
- 📐 Linear algebra view of ML models (vectors, dot products)
- 📉 Gradient-based learning and loss analysis
- 🛠️ Intentional feature engineering
- 🔎 SHAP-based explainability (local feature contributions)
- 🖥️ Interactive Streamlit dashboard for exploration & prediction

---
## ▶️ How to Run the App

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

The app includes:
- Dataset overview
- Exploratory analysis
- Model performance (ROC-AUC)
- Default risk prediction
- SHAP-based explanation of predictions

---

## 🧠 Explainability with SHAP

SHAP is used to break down each prediction into feature-level contributions:

- Positive SHAP value → increases default risk
- Negative SHAP value → decreases default risk
- Larger magnitude → stronger influence

This makes the model transparent, auditable, and stakeholder-friendly.


---

## ✨ Author

Built with intent, structure, and zero vibes-only ML.
