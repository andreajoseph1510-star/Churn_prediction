import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight",
    page_icon="📡",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_model()
model     = artifacts["model"]
scaler    = artifacts["scaler"]
le_dict   = artifacts["le_dict"]
cat_cols  = artifacts["cat_cols"]
feat_names = artifacts["feature_names"]

# ── Title ─────────────────────────────────────────────────────
st.title("📡 ChurnSight — Telecom Churn Predictor")
st.markdown("Predict whether a customer will churn · Understand why · Get AI advice")
st.divider()

st.sidebar.header("🧾 Enter Customer Details")

gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior         = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner        = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents     = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure         = st.sidebar.slider("Tenure (months)", 1, 72, 12)
phone          = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet       = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_sec     = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup  = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_prot    = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support   = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv   = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_mov  = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract       = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless      = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment        = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly        = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
total          = st.sidebar.number_input("Total Charges ($)", 18.0, 9000.0, 1500.0)

# ── Predict Button ────────────────────────────────────────────
if st.sidebar.button("🔍 Predict Churn", use_container_width=True):

    input_dict = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_sec,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_prot,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_mov,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
   }

    # Encode categoricals
    input_df = pd.DataFrame([input_dict])
    for col in cat_cols:
        if col in input_df.columns:
            le = le_dict[col]
            input_df[col] = le.transform(input_df[col])

    # Reorder columns
    input_df = input_df[feat_names]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    # ── Risk Badge ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{prob*100:.1f}%")

    with col2:
        if prob >= 0.70:
            st.error("🔴 HIGH RISK — Likely to Churn")
        elif prob >= 0.40:
            st.warning("🟡 MEDIUM RISK — Monitor Closely")
        else:
            st.success("🟢 LOW RISK — Likely to Stay")

    with col3:
        st.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")

    st.divider()

    # ── SHAP Explanation ──────────────────────────────────────
    st.subheader("🔍 Why did the model predict this?")
    st.caption("SHAP values show which factors pushed the prediction up (red) or down (blue)")

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.values[0],
            feature_names=feat_names
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()

    # ── Retention Tips ────────────────────────────────────────
    st.divider()
    st.subheader("💡 Retention Suggestions")

    if prob >= 0.70:
        st.markdown("""
        - 📞 Call the customer personally — offer a loyalty discount
        - 📋 Propose upgrading to a **One Year or Two Year contract**
        - 🎁 Offer a free month or service bundle upgrade
        """)
    elif prob >= 0.40:
        st.markdown("""
        - 📧 Send a satisfaction survey and follow up
        - 💰 Offer a small discount on monthly charges
        - 🔧 Check if they have any unresolved support issues
        """)
    else:
        st.markdown("""
        - ✅ Customer is stable — maintain regular engagement
        - 🎯 Good candidate for upselling premium services
        """)

else:
    st.info("👈 Fill in the customer details on the left and click **Predict Churn**")
