import streamlit as st
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt


from login import show_login_page
st.set_page_config(
    page_title="SenseChurn",
    page_icon="📡",
    layout="wide"
)
st.markdown("""
<style>

/* FORCE style for Logout button (last button in top row) */
div[data-testid="stHorizontalBlock"] div button {
    color: white !important;
}



</style>
""", unsafe_allow_html=True)
if "done" not in st.session_state:
    st.session_state.done = False



if "user" not in st.session_state:
    show_login_page()
    st.stop()
else:
    pass

#  Initialize FIRST (very important)
if "show_help" not in st.session_state:
    st.session_state.show_help = False
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False



top_bar = st.container()

with top_bar:
    col_title, col_space, col_help, col_feedback, col_logout = st.columns([6, 4, 1, 1, 1])

    with col_title:
        st.markdown("")

    with col_help:
        if st.button("❓", key="help_btn"):
            st.session_state.show_help = not st.session_state.show_help

    with col_feedback:
        if st.button("💬", key="feedback_btn"):
            st.session_state.show_feedback = not st.session_state.show_feedback

    with col_logout:
        if st.button("Logout", key="logout_btn"):
            st.session_state.clear()
            st.rerun()
st.markdown("""
<style>
/* Target ONLY the last container (Logout area) */
div[data-testid="column"]:last-child button {
    background-color: #00008B !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 8px !important;
}




/* Hover */
div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    box-shadow: 0 0 10px #8E2DE2, 0 0 20px #4A00E0;
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)





# Help popup
if st.session_state.show_help:
    st.info("""
    📘 Help Guide

    - Enter customer details  
    - View churn probability  
    - Ask AI questions  

    💡 Example: "How to reduce churn?"
    """)
# Feedback popup
if st.session_state.show_feedback:
    st.markdown("### 💬 Feedback")

    feedback = st.text_area("Write your feedback")

    if st.button("Submit Feedback"):
        if feedback.strip():
            with open("feedback.txt", "a") as f:
                f.write(feedback + "\n---\n")
            st.success("✅ Thank you!")
        else:
            st.warning("⚠️ Enter feedback")

    if st.button("Close"):
        st.session_state.show_feedback = False





st.markdown("""

<style>
@import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');
.stApp {
    background-color: black;
}
/* Hide Press Enter to apply - all methods */
[data-testid="InputInstructions"],
.st-emotion-cache-1rsyhoq,
small {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}


html, body, [class*="css"] {
    color: #e8e9f0;
    font-family: 'Roboto', sans-serif;
}


h1 {
    color: #818cf8;
    font-family: 'Audiowide', sans-serif !important;
    letter-spacing: 2px;
}


p {
    color: #22d3ee;
}



section[data-testid="stSidebar"] {
    background-color: #12142a;
}


section[data-testid="stSidebar"] label {
    color: #e8e9f0 !important;
}
.stButton > button {
    background-color: #1E3A8A;
    color: #000000;
    border-radius: 8px;
    border: none;
    font-weight: bold;
    transition: 0.3s ease;
}




.stButton > button:hover {
    background-color: #818cf8;
    color: black;
}

</style>
""", unsafe_allow_html=True)

import joblib
from xgboost import XGBClassifier

@st.cache_resource
def load_all():
    # Load model (JSON)
    model = XGBClassifier()
    model.load_model("model.json")

    # Load other artifacts
    artifacts = joblib.load("artifacts.pkl")

    return model, artifacts

model, artifacts = load_all()

scaler     = artifacts["scaler"]
le_dict    = artifacts["le_dict"]
cat_cols   = artifacts["cat_cols"]
feat_names = artifacts["feature_names"]

col1, col2 = st.columns([4, 3])
with col1:
    st.title("🔮 SenseChurn ")
    st.markdown("Telecom Churn Predictor")
    st.markdown("Predict whether a customer will churn · Understand why · Get AI advice")
with col2:
    with open("logo.gif", "rb") as f:
        import base64
        gif_data = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(
        f'<img src="data:image/gif;base64,{gif_data}" width="1500" height="200" style="border-radius:16px;">',
        unsafe_allow_html=True
    )


tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Insights", "🤖 AI Assistant"])

# ── Predict Button ────────────────────────────────────────────
with tab1:
    
    # ── Centered form ─────────────────────────────────────────
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.subheader("🧾 Enter Customer Details")
        st.markdown("---")

        gender         = st.selectbox("Gender", ["Male", "Female"])
        senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner        = st.selectbox("Has Partner", ["Yes", "No"])
        dependents     = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure         = st.slider("Tenure (months)", 1, 72, 12)
        phone          = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec     = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup  = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_prot    = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support   = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv   = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_mov  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment        = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

        col1, col2 = st.columns(2)
        with col1:
            monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0)
        with col2:
            total   = st.number_input("Total Charges ($)", 18.0, 9000.0, 1500.0)

        st.markdown("---")
        predict_btn = st.button("🔍 Predict Churn", use_container_width=True)

    # ── Results appear below centered ─────────────────────────
    if predict_btn:
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

        # Encode
        input_df = pd.DataFrame([input_dict])
        for col in cat_cols:
            if col in input_df.columns:
                le = le_dict[col]
                input_df[col] = le.transform(input_df[col])

        input_df    = input_df[feat_names]
        input_scaled = scaler.transform(input_df)

        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        

        # ← Add these two lines
        st.session_state["last_customer"] = input_dict
        st.session_state["last_prob"] = prob
        st.divider()
        

        # ── Risk Badge ────────────────────────────────────────
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

        # ── Feature Impact Chart ───────────────────────────────
        st.subheader("🔍 Why did the model predict this?")

        explainer    = shap.TreeExplainer(model)
        shap_values  = explainer.shap_values(input_scaled)

        impact_df = pd.DataFrame({
            "Feature": feat_names,
            "Impact": shap_values[0]
        }).sort_values("Impact", ascending=True)

        impact_df["Reason"] = impact_df["Impact"].apply(
            lambda x: "🔴 Pushes Towards Churn" if x > 0 else "🟢 Keeps Customer"
        )
        impact_df["Color"] = impact_df["Impact"].apply(
            lambda x: "#ff4757" if x > 0 else "#2ecc71"
        )
        impact_df = pd.concat([
            impact_df.tail(5),
            impact_df.head(5)
        ]).drop_duplicates()

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=impact_df["Impact"],
            y=impact_df["Feature"],
            orientation="h",
            marker_color=impact_df["Color"],
            text=impact_df["Reason"],
            textposition="outside"
        ))
        fig.update_layout(
            title="Top Factors Influencing This Prediction",
            paper_bgcolor="#12142a",
            plot_bgcolor="#12142a",
            font_color="#e8e9f0",
            title_font_color="#818cf8",
            height=400,
            xaxis_title="Impact on Churn Prediction",
            yaxis_title="Feature",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Plain English Summary ──────────────────────────────
        st.subheader("📋 Summary")
        churn_reasons = impact_df[impact_df["Impact"] > 0].sort_values("Impact", ascending=False).head(3)
        stay_reasons  = impact_df[impact_df["Impact"] < 0].sort_values("Impact").head(3)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔴 Top Churn Reasons:**")
            for _, row in churn_reasons.iterrows():
                st.markdown(f"- {row['Feature']}")
        with col2:
            st.markdown("**🟢 Top Staying Reasons:**")
            for _, row in stay_reasons.iterrows():
                st.markdown(f"- {row['Feature']}")

        st.divider()

        # ── Retention Suggestions ─────────────────────────────
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
            -  Customer is stable — maintain regular engagement
            -  Good candidate for upselling premium services
            """)
        
        # Save For download
        st.session_state["done"] = True
        st.session_state["prob"] = prob

        # create risk text (same logic)
        if prob >= 0.70:
            risk = "High Risk 🔴"
        elif prob >= 0.40:
            risk = "Medium Risk 🟡"
        else:
            risk = "Low Risk 🟢"

        st.session_state["risk"] = risk
    

    else:
        st.info("👆 Fill in the customer details above and click **Predict Churn**")
    
if st.session_state.get("done", False):

    prob = st.session_state["prob"]
    risk = st.session_state["risk"]
    customer = st.session_state.get("last_customer", {})

    report = f"""
CUSTOMER CHURN REPORT
====================================

Customer Details:
-----------------
Gender          : {customer.get("gender", "N/A")}
Contract Type   : {customer.get("Contract", "N/A")}
Payment Method  : {customer.get("PaymentMethod", "N/A")}

------------------------------------

Churn Probability : {prob:.2%}
Risk Level        : {risk}

------------------------------------
Generated by SenseChurn
"""

    st.download_button(
        "📥 Download Report",
        report,
        "SenseChurn_Report.txt"
    )

with tab2:
    import plotly.express as px
    import plotly.graph_objects as go

    df = pd.read_csv("telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    st.subheader("📊 Churn Insights Dashboard")
    st.markdown("Visual analysis of why customers churn")
    st.divider()

    # ── Row 1: Two charts side by side ────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        # Churn by Contract
        contract_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack()
        fig1 = px.bar(
            contract_churn,
            barmode="group",
            title="Churn Rate by Contract Type",
            color_discrete_map={"Yes": "#ff4757", "No": "#2ecc71"},
            template="plotly_dark"
        )
        fig1.update_layout(
            plot_bgcolor="#12142a",
            paper_bgcolor="#12142a",
            font_color="#e8e9f0",
            title_font_color="#818cf8"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Churn by Internet Service
        internet_churn = df.groupby("InternetService")["Churn"].value_counts(normalize=True).unstack()
        fig2 = px.bar(
            internet_churn,
            barmode="group",
            title="Churn Rate by Internet Service",
            color_discrete_map={"Yes": "#ff4757", "No": "#2ecc71"},
            template="plotly_dark"
        )
        fig2.update_layout(
            plot_bgcolor="#12142a",
            paper_bgcolor="#12142a",
            font_color="#e8e9f0",
            title_font_color="#818cf8"
        )
        st.plotly_chart(fig2, use_container_width=True)

   
    col3, col4 = st.columns(2)

    with col3:
        # Monthly Charges Distribution
        fig3 = px.histogram(
            df, x="MonthlyCharges",
            color="Churn",
            title="Monthly Charges — Churn vs No Churn",
            color_discrete_map={"Yes": "#ff4757", "No": "#2ecc71"},
            barmode="overlay",
            template="plotly_dark"
        )
        fig3.update_layout(
            plot_bgcolor="#12142a",
            paper_bgcolor="#12142a",
            font_color="#e8e9f0",
            title_font_color="#818cf8"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Churn by Tenure
        fig4 = px.histogram(
            df, x="tenure",
            color="Churn",
            title="Tenure — Churn vs No Churn",
            color_discrete_map={"Yes": "#ff4757", "No": "#2ecc71"},
            barmode="overlay",
            template="plotly_dark"
        )
        fig4.update_layout(
            plot_bgcolor="#12142a",
            paper_bgcolor="#12142a",
            font_color="#e8e9f0",
            title_font_color="#818cf8"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: Full width chart ────────────────────────────────
    # Payment Method Churn
    payment_churn = df.groupby("PaymentMethod")["Churn"].value_counts(normalize=True).unstack()
    fig5 = px.bar(
        payment_churn,
        barmode="group",
        title="Churn Rate by Payment Method",
        color_discrete_map={"Yes": "#ff4757", "No": "#2ecc71"},
        template="plotly_dark"
    )
    fig5.update_layout(
        plot_bgcolor="#12142a",
        paper_bgcolor="#12142a",
        font_color="#e8e9f0",
        title_font_color="#818cf8"
    )
    st.plotly_chart(fig5, use_container_width=True)

    # ── Row 4: Donut chart ─────────────────────────────────────
    col5, col6 = st.columns(2)
    with col5:
        churn_counts = df["Churn"].value_counts()
        fig6 = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Overall Churn Distribution",
            color_discrete_map={"Yes": "#ff4757", "No": "#2ecc71"},
            hole=0.4,
            template="plotly_dark"
        )
        fig6.update_layout(
            plot_bgcolor="#12142a",
            paper_bgcolor="#12142a",
            font_color="#e8e9f0",
            title_font_color="#818cf8"
        )
        st.plotly_chart(fig6, use_container_width=True)

    with col6:
        # Senior Citizen Churn
        senior_churn = df.groupby("SeniorCitizen")["Churn"].value_counts(normalize=True).unstack()
        senior_churn.index = ["Non-Senior", "Senior"]
        fig7 = px.bar(
            senior_churn,
            barmode="group",
            title="Churn Rate by Senior Citizen",
            color_discrete_map={"Yes": "#ff4757", "No": "#2ecc71"},
            template="plotly_dark"
        )
        fig7.update_layout(
            plot_bgcolor="#12142a",
            paper_bgcolor="#12142a",
            font_color="#e8e9f0",
            title_font_color="#818cf8"
        )
        st.plotly_chart(fig7, use_container_width=True)
with tab3:
    from recommender import get_retention_advice

    st.subheader("🤖 AI Retention Advisor")
    st.markdown("Get personalized retention strategies powered by AI")
    st.divider()

    if "last_customer" in st.session_state and "last_prob" in st.session_state:
        customer = st.session_state["last_customer"]
        prob     = st.session_state["last_prob"]

        st.markdown(f"**Current customer churn probability: `{prob*100:.1f}%`**")
        st.divider()

        #  Auto advice 
        if st.button("🎯 Get AI Retention Advice", use_container_width=True):
            with st.spinner("Thinking..."):
                advice = get_retention_advice(customer, prob)
                st.session_state["ai_advice"] = advice  # ← save to session

        if "ai_advice" in st.session_state:
            st.markdown(st.session_state["ai_advice"])

        st.divider()

        # Chat 
        st.markdown("#### 💬 Ask a specific question")
        question = st.text_input("e.g. Should I offer a discount or upgrade?",
                                  key="ai_question")

        if st.button("Ask AI", use_container_width=True, key="ask_btn"):
            if question:
                with st.spinner("Thinking..."):
                    answer = get_retention_advice(customer, prob, question)
                    st.session_state["ai_answer"] = answer  # ← save to session
            else:
                st.warning("Please type a question first!")

        if "ai_answer" in st.session_state:
            st.markdown(st.session_state["ai_answer"])

    else:
        st.info("👆 First go to the **Predict tab**, enter customer details and click Predict — then come back here!")

