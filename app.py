import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Attack Risk Predictor",
                   page_icon="❤️",
                   layout="wide")

pipeline = joblib.load("heart_pipeline.pkl")
feature_names = joblib.load("feature_names.pkl")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("📊 Model Insights")

st.sidebar.markdown("### 🔬 Most Influential Features")

st.sidebar.markdown("""
**1️⃣ ST Depression (Oldpeak)**  
⬆ Higher value → Higher risk  

**2️⃣ Chest Pain Type (Atypical/Non-anginal)**  
Certain types reduce risk  

**3️⃣ Thalassemia (Reversible Defect)**  
⬆ Associated with higher risk  

**4️⃣ Slope of ST Segment (Flat)**  
⬆ Indicates ischemia  

**5️⃣ Sex (Male)**  
⬆ Slightly higher risk  

**6️⃣ Exercise Induced Angina**  
⬆ Strong predictor  

""")

st.sidebar.markdown("---")
st.sidebar.caption("Feature importance derived from logistic regression coefficients.")

# ---------------- MAIN HEADER ---------------- #
st.markdown("<h1 style='text-align:center;'>❤️ Heart Attack Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; font-size:14px; color:#888; margin-top:-10px;'>
📱 <b>Mobile Users:</b> Tap the top-left menu icon (☰) to view model insights and feature importance.
</div>
""", unsafe_allow_html=True)
st.markdown("---")


col1, col2 = st.columns(2)

# ---------------- USER INPUT (Readable Labels) ---------------- #
with col1:
    age = st.number_input("Age", 20, 100, 50)

    sex = st.selectbox("Sex", ["Female", "Male"])

    cp = st.selectbox("Chest Pain Type", {
        "Typical Angina (Low Risk)": 1,
        "Atypical Angina": 2,
        "Non-Anginal Pain": 3,
        "Asymptomatic (High Risk)": 4
    })

    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL",
                       {"No": 0, "Yes": 1})

with col2:
    restecg = st.selectbox("Resting ECG Result", {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    })

    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)

    exang = st.selectbox("Exercise Induced Angina",
                         {"No": 0, "Yes": 1})

    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

    slope = st.selectbox("Slope of Peak Exercise ST Segment", {
        "Upsloping (Lower Risk)": 1,
        "Flat (Higher Risk)": 2,
        "Downsloping (High Risk)": 3
    })

    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0,1,2,3])

    thal = st.selectbox("Thalassemia Type", {
        "Normal": 3,
        "Fixed Defect": 6,
        "Reversible Defect (Higher Risk)": 7
    })

st.markdown("---")

# Convert selections
sex = 1 if sex == "Male" else 0
cp = cp[list(cp.keys())[0]] if isinstance(cp, dict) else cp
fbs = fbs[list(fbs.keys())[0]] if isinstance(fbs, dict) else fbs
restecg = restecg[list(restecg.keys())[0]] if isinstance(restecg, dict) else restecg
exang = exang[list(exang.keys())[0]] if isinstance(exang, dict) else exang
slope = slope[list(slope.keys())[0]] if isinstance(slope, dict) else slope
thal = thal[list(thal.keys())[0]] if isinstance(thal, dict) else thal

# ---------------- PREDICTION ---------------- #
if st.button("🔍 Predict Risk", use_container_width=True):

    input_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)

    for col in feature_names:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[feature_names]

    probability = pipeline.predict_proba(input_df)[0][1]

    st.markdown("## 📊 Risk Assessment")

    if probability < 0.30:
        st.success(f"🟢 Low Risk ({probability:.2%})")
    elif probability < 0.60:
        st.warning(f"🟡 Moderate Risk ({probability:.2%})")
    else:
        st.error(f"🔴 High Risk ({probability:.2%})")

    st.progress(int(probability * 100))

st.markdown("---")
st.caption("⚠ Educational tool only. Not a medical diagnostic system.")