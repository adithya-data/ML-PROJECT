import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#Web Page Design
st.set_page_config(page_title="Women Productivity Predictor", page_icon="🌸", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

* {
    font-family: 'Outfit', sans-serif;
}

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #fff5f8 0%, #ffe4ec 100%);
}

/* Header Styling */
h1 {
    color: #880e4f !important;
    font-weight: 600;
    text-align: center;
    margin-bottom: 30px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

h3 {
    color: #000000 !important;
    font-weight: 600;
    margin-top: 20px;
}

/* Card-like containers for inputs */
div[data-testid="stVerticalBlock"] > div {
    background: rgba(255, 255, 255, 0.4);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 10px;
}

/* Input widget styling */
div[data-testid="stNumberInput"] input {
    background-color: white !important;
    color: #000000 !important;
    border: 1px solid #f48fb1 !important;
    border-radius: 10px;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #000000 !important;
    border: 1px solid #f48fb1 !important;
    border-radius: 10px;
}

/* BaseWeb Select text color */
div[data-baseweb="select"] * {
    color: #000000 !important;
}

/* Slider Customization */
.stSlider {
    padding-bottom: 20px;
}

/* Button Styling */
div.stButton > button {
    background: linear-gradient(90deg, #880e4f 0%, #ad1457 100%);
    color: white !important;
    border-radius: 25px;
    height: 3.5em;
    width: 100%;
    font-size: 18px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(136, 14, 79, 0.3);
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(136, 14, 79, 0.4);
    color: #ffd6e7 !important;
}

/* Result Message Styling */
.stAlert {
    border: none !important;
    background: rgba(255, 192, 203, 0.2) !important;
    border-radius: 15px !important;
    border-left: 5px solid #880e4f !important;
}

/* Label and Markdown Text Visibility */
div[data-testid="stMarkdownContainer"] p, 
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stVerticalBlock"] label {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Divider Styling */
hr {
    border-top: 1px solid #f48fb1 !important;
    margin: 40px 0 !important;
}

</style>
""", unsafe_allow_html=True)



#Model Loading
with open("productivity_model.pkl", "rb") as f:
    artifact = pickle.load(f) 

model = artifact["model"]
FEATURES = artifact["features"]
scaler = artifact.get("scaler")

# -----------------------------
# Page title
# -----------------------------
st.title("🌸Women Productivity Prediction")
st.image("cycle_sync.jpg", width=500)

st.write("Lets check your productivity level!!")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("🗓️ Cycle Information")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**📅 Cycle Length (days)**")
    cycle_length = st.number_input("Cycle Length", 20, 40, 28, label_visibility="collapsed")
    
    st.markdown("**🩸 Flow Level (1-10)**")
    flow_level = st.slider("Flow Level", 1, 10, 5, label_visibility="collapsed")

with col2:
    st.markdown("**🔄 Current Cycle Phase**")
    cycle_phase = st.selectbox(
        "Current Cycle Phase",
        ["Follicular", "Luteal", "Menstrual"],
        label_visibility="collapsed"
    )
    
    st.markdown("**🥚 Ovulation Test Result**")
    ovulation = st.selectbox(
        "Ovulation Test Result",
        ["Positive", "Negative"],
        label_visibility="collapsed"
    )

st.divider()

st.subheader("🧠 Wellbeing & Symptoms")
col3, col4 = st.columns(2)

with col3:
    st.markdown("**😟 Pain Level (0-10)**")
    pain_level = st.slider("Pain Level", 0, 10, 1, label_visibility="collapsed")
    
    st.markdown("**🥨 PMS Symptoms**")
    pms = st.selectbox("PMS Symptoms", ["No", "Yes"], label_visibility="collapsed")
    
    st.markdown("**😊 Mood Score (1-10)**")
    mood = st.slider("Mood Score", 1, 10, 6, label_visibility="collapsed")

with col4:
    st.markdown("**😫 Stress Level (1-10)**")
    stress = st.number_input("Stress Level", 1.0, 10.0, 5.0, 0.1, label_visibility="collapsed")
    
    st.markdown("**⚡ Energy Level (1-10)**")
    energy = st.slider("Energy Level", 1, 10, 6, label_visibility="collapsed")
    
    st.markdown("**🧠 Concentration Level (1-10)**")
    concentration = st.slider("Concentration Level", 1, 10, 6, label_visibility="collapsed")

st.divider()

st.subheader("😴 Sleep & Work")
col5, col6 = st.columns(2)

with col5:
    st.markdown("**😴 Sleep Hours**")
    sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0, label_visibility="collapsed")
    
    st.markdown("**🏥 PCOS Diagnosed**")
    pcos = st.selectbox("PCOS Diagnosed", ["No", "Yes"], label_visibility="collapsed")

with col6:
    st.markdown("**📉 Work Hours Lost**")
    work_lost = st.number_input("Work Hours Lost", 0.0, 10.0, 1.0, label_visibility="collapsed")

st.divider()

# Optional hormone values
st.subheader("🧪 Hormone Levels (Optional)")
col7, col8 = st.columns(2)

with col7:
    st.markdown("**💉 Estrogen (pg/ml)** *(typical: 50–200)*")
    estrogen = st.number_input("Estrogen", 0.0, 500.0, 90.0, label_visibility="collapsed")

with col8:
    st.markdown("**💉 Progesterone (ng/ml)** *(typical: 0.5–20)*")
    progesterone = st.number_input("Progesterone", 0.0, 50.0, 4.5, label_visibility="collapsed")

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

    # Scale inputs using the fitted scaler from the artifact
    if scaler is not None:
        input_data_scaled = scaler.transform(input_df)
        prediction = model.predict(input_data_scaled)
    else:
        prediction = model.predict(input_df.values)

    # Cluster mapping from notebook: 0=Low, 1=Medium, 2=High
    cluster_map = {0: "Low, you need rest 💤", 1: "Medium, have a good day 🌤️", 2: "High, you are active today 🚀"}
    label = cluster_map.get(int(prediction[0]), "Unknown")
    st.success(f"🌸 Productivity Level: **{label}**")