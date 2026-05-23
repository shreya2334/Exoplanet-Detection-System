import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Exoplanet Detector",
    page_icon="🪐",
    layout="wide"
)

# Custom Styling

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0E1117;
        color: white;
    }

    h1, h2, h3 {
        color: #7FDBFF;
    }

    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("🪐 Exoplanet Detection System")
st.write("Machine Learning powered exoplanet prediction system")

st.sidebar.title("🪐 About")

st.sidebar.info(
    """
    This application predicts whether
    a detected astronomical signal
    is likely to be a real exoplanet
    or a false positive.

    Models Used:
    - XGBoost
    - Random Forest
    - CNN
    - LSTM
    """
)

st.sidebar.success("Exploring Worlds with AI")

model = joblib.load("models/ml/xgboost.pkl")
scaler = joblib.load("models/ml/scaler.pkl")

st.header("Enter Exoplanet Parameters")

col1, col2 = st.columns(2)

with col1:

    koi_period = st.number_input(
        "Orbital Period",
        min_value=0.0,
        value=10.0
    )

    koi_prad = st.number_input(
        "Planet Radius",
        min_value=0.0,
        value=2.0
    )

    koi_depth = st.number_input(
        "Transit Depth",
        min_value=0.0,
        value=500.0
    )

    koi_duration = st.number_input(
        "Transit Duration",
        min_value=0.0,
        value=5.0
    )

    koi_impact = st.number_input(
        "Impact Parameter",
        min_value=0.0,
        value=0.5
    )

with col2:

    koi_insol = st.number_input(
        "Insolation Flux",
        min_value=0.0,
        value=100.0
    )

    koi_teq = st.number_input(
        "Equilibrium Temperature",
        min_value=0.0,
        value=500.0
    )

    koi_steff = st.number_input(
        "Stellar Effective Temperature",
        min_value=0.0,
        value=5500.0
    )

    koi_srad = st.number_input(
        "Stellar Radius",
        min_value=0.0,
        value=1.0
    )

    koi_model_snr = st.number_input(
        "Model Signal-to-Noise Ratio",
        min_value=0.0,
        value=20.0
    )

if st.button("Predict Exoplanet"):
    input_data = pd.DataFrame({

        'koi_period': [koi_period],
        'koi_prad': [koi_prad],
        'koi_depth': [koi_depth],
        'koi_duration': [koi_duration],

        'koi_impact': [koi_impact],
        'koi_insol': [koi_insol],
        'koi_teq': [koi_teq],

        'koi_steff': [koi_steff],
        'koi_srad': [koi_srad],

        'koi_model_snr': [koi_model_snr]
    })

    st.write("Input Data:")
    st.dataframe(input_data)

    scaled_input = scaler.transform(input_data)

    scaled_input = pd.DataFrame(
                        scaled_input, 
                        columns= input_data.columns
                    )
    
    prediction_proba = model.predict_proba(scaled_input)
    st.subheader("Prediction Result")

    exoplanet_probability = prediction_proba[0][1]

    st.write(f"Raw Exoplanet Probability: {exoplanet_probability:.4f}")

    if exoplanet_probability >= 0.90:
        st.success("High Confidence Exoplanet Candidate")
        st.balloons()

    elif exoplanet_probability >= 0.70:
        st.info("Moderate Probability Exoplanet Signal")

    else:
        st.warning("Signal resembles a False Positive")

    confidence = max(prediction_proba[0])

    st.write(f"Prediction Confidence: {confidence:.4f}")
    st.progress(float(confidence))

st.markdown("---")

st.subheader("Model Information")

st.write("""
This prediction is generated using an improved XGBoost classifier
trained on processed NASA Kepler exoplanet data.

Input Features Used:
- Orbital Period
- Planet Radius
- Transit Depth
- Transit Duration
- Impact Parameter
- Insolation Flux
- Equilibrium Temperature
- Stellar Effective Temperature
- Stellar Radius
- Model Signal-to-Noise Ratio

The model analyzes planetary transit patterns and stellar
characteristics to estimate whether the detected signal
resembles a potential exoplanet candidate or a likely
false positive.
""")

st.markdown("---")
st.caption("🚀Exploring distant worlds with Artificial Intelligence🚀")