import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#Web Page Desing
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color:pink;
    }

    /* Title */
        h1{
            color:blue;
            }
            
    

/* Description text */
p {
    color: white;
    font-weight: 500;
}

/* Subheaders like "Hormone Levels (Optional)" */
h3 {
    color: #f48fb1;
}
    }

    /* Labels */
    label {
        color: #4a4a4a !important;
        font-weight: 500;
    }

    /* Buttons */
    div.stButton > button {
        background-color:#f48fb1;
        color: pink;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #f48fb1;
        color: pink;
    }

    /* Success box */
    .stAlert > div {
        background-color:#444444;
        color: #880e4f;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


#Model Loading
with open("productivity_model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
FEATURES = artifact["features"]
# -----------------------------
# Model feature list (MUST match training)
# -----------------------------
FEATURES = [
    'cycle_length_days', 'flow_level', 'pain_level', 'pms_symptoms',
    'mood_score', 'stress_score_cycle', 'energy_level',
    'concentration_score', 'work_hours_lost', 'estrogen_pgml',
    'progesterone_ngml', 'sleep_hours', 'pcos_diagnosed',
    'cycle_phase_Follicular', 'cycle_phase_Luteal', 'cycle_phase_Menstrual',
    'ovulation_result_Negative', 'ovulation_result_Positive'
]

# -----------------------------
# Page title
# -----------------------------
st.title("Women Productivity Prediction")
st.write("Lets check your productivity level!!")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
cycle_length = st.number_input("Cycle Length (days)", 20, 40, 28)
flow_level = st.slider("Flow Level", 1, 5, 3)
pain_level = st.slider("Pain Level", 0, 10, 3)
pms = st.selectbox("PMS Symptoms", ["No", "Yes"])

mood = st.slider("Mood Score", 1, 10, 6)
stress = st.slider("Stress Level", 1, 10, 5)
energy = st.slider("Energy Level", 1, 10, 6)
concentration = st.slider("Concentration Level", 1, 10, 6)

sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
work_lost = st.number_input("Work Hours Lost", 0.0, 10.0, 1.0)

pcos = st.selectbox("PCOS Diagnosed", ["No", "Yes"])

cycle_phase = st.selectbox(
    "Current Cycle Phase",
    ["Follicular", "Luteal", "Menstrual"]
)

ovulation = st.selectbox(
    "Ovulation Test Result",
    ["Positive", "Negative"]
)

st.divider()

# Optional hormone values
st.subheader("Hormone Levels")

estrogen = st.number_input("Estrogen (pg/ml)", 0.0, 500.0, 0.0)
progesterone = st.number_input("Progesterone (ng/ml)", 0.0, 50.0, 0.0)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Productivity"):

    # Create zero-filled feature dict
    input_data = dict.fromkeys(FEATURES, 0)

    # Fill numeric features
    input_data['cycle_length_days'] = cycle_length
    input_data['flow_level'] = flow_level
    input_data['pain_level'] = pain_level
    input_data['pms_symptoms'] = 1 if pms == "Yes" else 0
    input_data['mood_score'] = mood
    input_data['stress_score_cycle'] = stress
    input_data['energy_level'] = energy
    input_data['concentration_score'] = concentration
    input_data['sleep_hours'] = sleep
    input_data['work_hours_lost'] = work_lost
    input_data['pcos_diagnosed'] = 1 if pcos == "Yes" else 0
    input_data['estrogen_pgml'] = estrogen
    input_data['progesterone_ngml'] = progesterone

    # One-hot encoding
    input_data[f'cycle_phase_{cycle_phase}'] = 1
    input_data[f'ovulation_result_{ovulation}'] = 1

    # Convert to DataFrame (ORDER MATTERS)
    input_df = pd.DataFrame([input_data])[FEATURES]

    # Prediction
    prediction = model.predict(input_df)

    st.success(f"Predicted Productivity Score: {prediction[0]}")
    cluster_map = {0: "Low, you need rest", 1: "Medium, have a good day", 2: "High, you are active today"}
    label = cluster_map.get(int(prediction[0]), "Unknown")
    st.success(f"🌸 Productivity Level: **{label}**")