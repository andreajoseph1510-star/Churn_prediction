import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from shap.plots._waterfall import waterfall_legacy

#  Page configuration
st.set_page_config(
    page_title="SenseChurn",
    page_icon="📡",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background-color: black;
}


html, body, [class*="css"] {
    color: #e8e9f0;
    font-family: 'Roboto', sans-serif;
}


h1 {
    color: #818cf8;
    font-family: 'Space Mono', monospace;
}


p {
    color: #a0a3c4;
}


section[data-testid="stSidebar"] {
    background-color: #12142a;
}


section[data-testid="stSidebar"] label {
    color: #e8e9f0 !important;
}


.stButton > button {
    background-color: #4f46e5;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #818cf8;
    color: black;
}

</style>
""", unsafe_allow_html=True)

#  Load model 
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


tab1, tab2 = st.tabs(["🔍 Predict", "📊 Insights"])
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
with tab1:
    if st.sidebar.button("🔮 Predict Churn", use_container_width=True):

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

        # Risk Badge
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Churn Probability", f"{prob*100:.1f}%")

        with col2:
            if prob >= 0.70:
                st.error("🔴 HIGH RISK — Likely to Churn")
            elif prob >= 0.40:
                st.warning("🟡 MEDIUM RISK — Monitor Closely")
            else:
                st.success("🟢 LOW RISK — Most Likely to Stay")

        with col3:
            st.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")

        st.divider()

        # ── SHAP Explanation ──────────────────────────────────────
        # st.subheader(" Why did the model predict this?")
        # st.caption("The factors which  pushed the prediction up (red) or down (blue)")
        # fig, ax = plt.subplots(figsize=(10, 4))

        # # Set background color
        # fig.patch.set_facecolor("#170740")   # whole figure
        # ax.set_facecolor("#06d4d4f6")          # plot area

        # explainer   = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(input_scaled)

        # fig, ax = plt.subplots(figsize=(10, 4))

        # fig.patch.set_facecolor("#020f17")
        # ax.set_facecolor("#13dbd7")

        # shap.waterfall_plot(
        #     shap.Explanation(
        #         values=shap_values[0],  # or shap_values[0][0] if error persists
        #         base_values=explainer.expected_value,
        #         data=input_df.values[0],
        #         feature_names=feat_names
        #     ),
        #     show=False
        # )
        import matplotlib.pyplot as plt
        import shap

        # 🌙 Set global dark theme (THIS is the key fix)
        plt.rcParams.update({
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white'
        })

        # ── Feature Impact Chart ───────────────────────────────────
        st.subheader("🔍 Why did the model predict this?")

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        # Build impact dataframe
        impact_df = pd.DataFrame({
            "Feature": feat_names,
            "Impact": shap_values[0]
        }).sort_values("Impact", ascending=True)

        # Label as churn reason or staying reason
        impact_df["Reason"] = impact_df["Impact"].apply(
            lambda x: "🔴 Pushes Towards Churn" if x > 0 else "🟢 Keeps Customer"
        )
        impact_df["Color"] = impact_df["Impact"].apply(
            lambda x: "#ff4757" if x > 0 else "#2ecc71"
        )

        # Only show top 5 most impactful features
        impact_df = pd.concat([
            impact_df.tail(5),   # top 5 staying reasons
            impact_df.head(5)    # top 5 churn reasons
        ]).drop_duplicates()

        # Plot
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

    # ── Row 2: Two charts side by side ────────────────────────
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
