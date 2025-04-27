# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from io import BytesIO

# --- Page config ---
st.set_page_config(page_title="Churn Prediction Ultimate", page_icon="🔮", layout="wide")

# --- Custom CSS Styling for Apple Look ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'DM Sans', sans-serif;
        background-color: #f5f5f7;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #1d1d1f;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 22px;
        color: #4b4b4f;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model and scaler ---
rf = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

st.markdown('<p class="title" style="color: #F8F8F8;">🔮 Customer Churn Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle" style="color: #F8F8F8;">Predict and prevent customer loss with intelligent machine learning models</p>', unsafe_allow_html=True)
st.divider()

# --- Sidebar Settings ---
st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider("Set Churn Probability Threshold", 0.1, 0.9, 0.25, step=0.01)
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit, ML & Creativity ✨")

# --- Tabs for Single vs Batch Prediction ---
tab1, tab2 = st.tabs(["🧍 Single Prediction", "📄 Batch Prediction"])

# --- Single Customer Prediction ---
with tab1:
    st.subheader("🧍 Single Customer Prediction")

    with st.form("single_customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            days_since_last_order = st.number_input("Days Since Last Order", 0, 365, 30)
            order_count = st.number_input("Order Count", 0, 50, 5)
            cashback_amount = st.number_input("Cashback Amount", 0.0, 10000.0, 100.0)

        with col2:
            complain = st.selectbox("Complain", [0, 1])
            satisfaction_score = st.slider("Satisfaction Score", 1, 5, 4)
            devices = st.number_input("Devices Registered", 1, 10, 2)
            warehouse_to_home = st.number_input("Warehouse to Home Distance (km)", 0, 1000, 20)

        predict_single = st.form_submit_button("Predict Churn")

    if predict_single:
        with st.spinner('Predicting... 🔮'):
            input_df = pd.DataFrame({
                'Tenure': [tenure],
                'DaySinceLastOrder': [days_since_last_order],
                'OrderCount': [order_count],
                'CashbackAmount': [cashback_amount],
                'Complain': [complain],
                'SatisfactionScore': [satisfaction_score],
                'NumberOfDeviceRegistered': [devices],
                'WarehouseToHome': [warehouse_to_home]
            })

            scaled_input = scaler.transform(input_df)
            churn_prob = rf.predict_proba(scaled_input)[:, 1][0]

        st.success(f"Churn Probability: **{churn_prob:.2f}**")

        if churn_prob >= threshold:
            st.error("❌ Prediction: Customer is likely to churn!")
        else:
            st.success("✅ Prediction: Customer is likely to stay.")

        # Gauge Meter
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            title={'text': "Churn Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'bar': {'color': "red" if churn_prob >= threshold else "green"}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# --- Batch Customer Prediction ---
with tab2:
    st.subheader("📄 Batch Customer Prediction")

    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("Uploaded Data Preview:")
        st.dataframe(data.head(), use_container_width=True)

        required_columns = [
            'Tenure', 'DaySinceLastOrder', 'OrderCount', 'CashbackAmount',
            'Complain', 'SatisfactionScore', 'NumberOfDeviceRegistered', 'WarehouseToHome'
        ]

        missing = [col for col in required_columns if col not in data.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            with st.spinner('Predicting batch... 📄'):
                scaled_batch = scaler.transform(data[required_columns])
                churn_probs = rf.predict_proba(scaled_batch)[:, 1]

                data['Churn_Probability'] = churn_probs
                data['Prediction'] = np.where(churn_probs >= threshold, 'Churn', 'Stay')

            st.success("✅ Batch Predictions Completed!")
            st.dataframe(data, use_container_width=True)

            # Pie Chart
            fig_pie = px.pie(data, names='Prediction', title='Churn vs Stay Distribution',
                             color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)

            # If Actual Churn labels are available
            if 'ActualChurn' in data.columns:
                y_true = data['ActualChurn']
                y_pred = np.where(data['Churn_Probability'] >= threshold, 1, 0)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
                col2.metric("Precision", f"{precision_score(y_true, y_pred):.2f}")
                col3.metric("Recall", f"{recall_score(y_true, y_pred):.2f}")
                col4.metric("F1 Score", f"{f1_score(y_true, y_pred):.2f}")

                cm = confusion_matrix(y_true, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)

            # Download Button
            towrite = BytesIO()
            downloaded_file = data.to_csv(index=False)
            towrite.write(downloaded_file.encode())
            towrite.seek(0)
            st.download_button(
                label="📥 Download Predictions",
                data=towrite,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )

# --- Footer ---
st.divider()
st.markdown("<center>Built with ❤️ using Streamlit | </center>", unsafe_allow_html=True)
