import streamlit as st
import pandas as pd
from joblib import load

# Load model & columns
model = load("salary_model.pkl")
model_columns = load("model_columns.pkl")

# Page config
st.set_page_config(page_title="Salary Predictor", layout="centered")

# 🧠 Title section
st.title("💰 Salary Predictor")
st.caption("🚀 Enter your details and get an instant salary estimate")

st.divider()

# 📥 Input section (better alignment using columns)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=18, max_value=90, step=1)
    experience = st.number_input("💼 Experience (years)", min_value=0, max_value=70, step=1)

with col2:
    hours = st.number_input("⏱ Hours per week", min_value=20, max_value=80, step=1)
    education = st.selectbox("🎓 Education Level", ["Bachelors", "Masters", "PhD"])

st.divider()

# 🚀 Centered button
_, center_col, _ = st.columns([1,2,1])
with center_col:
    predict_clicked = st.button("🚀 Predict Salary", use_container_width=True)

# 🔥 Prediction logic
if predict_clicked:

    # Input processing
    input_data = {
        'age': age,
        'experience': experience,
        'hours_per_week': hours,
        'education_level_Masters': 0,
        'education_level_PhD': 0
    }

    if education == "Masters":
        input_data['education_level_Masters'] = 1
    elif education == "PhD":
        input_data['education_level_PhD'] = 1

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.divider()

    # 💵 Result display
    st.success(f"💵 Estimated Salary: ${prediction:,.2f} / year")
    st.caption("⚠️ This is an estimate. Unusual inputs may reduce accuracy.")

st.divider()

st.caption("Built with ❤️ using Machine Learning")