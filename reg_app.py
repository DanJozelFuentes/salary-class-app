import streamlit as st
import joblib
import numpy as np
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# ================= MODEL PATH =================
MODEL_PATH = "salary_model.pkl"

# ================= LOAD MODEL SAFELY =================
if not os.path.exists(MODEL_PATH):
    st.error("❌ salary_model.pkl not found in this folder.")
    st.info("Make sure the file is in the same directory as reg_app.py")
    st.stop()

model = joblib.load(MODEL_PATH)

# ================= UI HEADER =================
st.title("💼 AI Salary Prediction System")
st.write("Predict salary based on years of experience using Machine Learning.")

st.markdown("---")

# ================= INPUT SECTION (FIXED) =================
years_experience = st.slider(
    "Years of Experience",
    min_value=0,
    max_value=50,
    value=1,
    step=1
)

st.info(f"Selected Experience: {years_experience} year(s)")

# ================= PREDICTION =================
if st.button("Predict Salary"):

    try:
        input_data = np.array([[years_experience]])
        prediction = model.predict(input_data)

        st.success(f"💰 Estimated Salary: ₱{prediction[0]:,.2f}")

    except Exception as e:
        st.error("Prediction failed. Check your model or input.")
        st.code(str(e))