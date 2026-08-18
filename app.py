import streamlit as st
import pandas as pd
import joblib

try:
    import shap
except ImportError:
    shap = None


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CardioRisk Assessment",
    page_icon="♡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #F4F0E7 !important;
    color: #493C32 !important;
}

.block-container {
    max-width: 1400px !important;
    padding-top: 2rem !important;
    padding-bottom: 5rem !important;
}

p {
    color: #5C5147 !important;
}

label {
    color: #5C5147 !important;
}

h1 {
    color: #493C32 !important;
    font-family: Georgia, serif !important;
    font-size: 52px !important;
    font-weight: 500 !important;
}

h2 {
    color: #493C32 !important;
    font-family: Georgia, serif !important;
    font-size: 38px !important;
    font-weight: 500 !important;
}

h3 {
    color: #493C32 !important;
    font-family: Georgia, serif !important;
    font-weight: 500 !important;
}

.section-label {
    color: #74856A !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 4px !important;
    margin-top: 28px !important;
    margin-bottom: 8px !important;
}


/* =========================================================
   INPUT LABELS
========================================================= */

.stNumberInput label,
.stSelectbox label,
.stTextInput label {
    color: #62564B !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}


/* =========================================================
   NUMBER INPUT
========================================================= */

.stNumberInput > div {
    background-color: #FBF9F4 !important;
    border-radius: 13px !important;
}

.stNumberInput [data-baseweb="input"] {
    background-color: #FBF9F4 !important;
    border: 1px solid #D7CCBB !important;
    border-radius: 13px !important;
}

.stNumberInput input {
    background-color: #FBF9F4 !important;
    color: #2F2924 !important;
    -webkit-text-fill-color: #2F2924 !important;
    caret-color: #2F2924 !important;
    font-size: 17px !important;
    font-weight: 500 !important;
}

.stNumberInput input::placeholder {
    color: #6B6055 !important;
    opacity: 1 !important;
}

.stNumberInput button {
    background-color: #E6DFD3 !important;
    color: #2F2924 !important;
    border: none !important;
}

.stNumberInput button span,
.stNumberInput button svg {
    color: #2F2924 !important;
    fill: #2F2924 !important;
}


/* =========================================================
   SELECTBOX
========================================================= */

.stSelectbox [data-baseweb="select"] {
    background-color: #FBF9F4 !important;
    color: #2F2924 !important;
}

.stSelectbox [data-baseweb="select"] > div {
    background-color: #FBF9F4 !important;
    border: 1px solid #D7CCBB !important;
    border-radius: 13px !important;
    min-height: 48px !important;
}

.stSelectbox [data-baseweb="select"] *,
.stSelectbox [data-baseweb="select"] div,
.stSelectbox [data-baseweb="select"] span {
    color: #2F2924 !important;
    -webkit-text-fill-color: #2F2924 !important;
    opacity: 1 !important;
}

.stSelectbox [data-baseweb="select"] svg {
    fill: #493C32 !important;
    color: #493C32 !important;
}


/* =========================================================
   DROPDOWN
========================================================= */

[data-baseweb="popover"] {
    background-color: #FBF9F4 !important;
    border: 1px solid #D7CCBB !important;
}

[data-baseweb="menu"] {
    background-color: #FBF9F4 !important;
}

[role="option"] {
    background-color: #FBF9F4 !important;
    color: #2F2924 !important;
    -webkit-text-fill-color: #2F2924 !important;
}

[role="option"] * {
    color: #2F2924 !important;
    -webkit-text-fill-color: #2F2924 !important;
    opacity: 1 !important;
}

[role="option"]:hover {
    background-color: #E7EDE1 !important;
    color: #2F2924 !important;
}

[role="option"][aria-selected="true"] {
    background-color: #E7EDE1 !important;
    color: #2F2924 !important;
}


/* =========================================================
   BUTTON
========================================================= */

.stButton > button {
    background-color: #74856A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 14px !important;
    min-height: 55px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background-color: #617257 !important;
    color: #FFFFFF !important;
}


/* =========================================================
   DIVIDERS
========================================================= */

hr {
    border: none !important;
    border-top: 1px solid #D7CCBB !important;
    margin-top: 35px !important;
    margin-bottom: 35px !important;
}


/* =========================================================
   RESULT CARDS
========================================================= */

.result-card {
    background-color: #FBF9F4 !important;
    border: 1px solid #D7CCBB !important;
    border-radius: 20px !important;
    padding: 25px !important;
    min-height: 145px !important;
}

.result-label {
    color: #74856A !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin-bottom: 12px !important;
}

.result-value {
    color: #493C32 !important;
    font-family: Georgia, serif !important;
    font-size: 32px !important;
    font-weight: 500 !important;
}


/* =========================================================
   INFORMATION BOX
========================================================= */

.insight-box {
    background-color: #E6EDE1 !important;
    border-left: 5px solid #74856A !important;
    border-radius: 14px !important;
    padding: 20px 25px !important;
    margin: 25px 0 !important;
    color: #493C32 !important;
    font-size: 17px !important;
}


/* =========================================================
   STREAMLIT METRICS
========================================================= */

[data-testid="stMetric"] {
    background-color: #FBF9F4 !important;
    border: 1px solid #D7CCBB !important;
    border-radius: 18px !important;
    padding: 20px !important;
}

[data-testid="stMetricLabel"] {
    color: #74856A !important;
}

[data-testid="stMetricValue"] {
    color: #493C32 !important;
}


/* =========================================================
   DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {
    border: 1px solid #D7CCBB !important;
    border-radius: 14px !important;
}


/* =========================================================
   ALERTS
========================================================= */

.stAlert {
    border-radius: 14px !important;
}


/* =========================================================
   HIDE STREAMLIT DEFAULT MENU
========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* =========================================================
   FORCE TEXT VISIBILITY
========================================================= */

.stSelectbox,
.stNumberInput {
    color: #2F2924 !important;
}

.stSelectbox *,
.stNumberInput * {
    --text-color: #2F2924;
}


/* =========================================================
   MOBILE
========================================================= */

@media (max-width: 900px) {

    h1 {
        font-size: 40px !important;
    }

    h2 {
        font-size: 30px !important;
    }

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

}


/* =========================================================
   SHAP
========================================================= */

.shap-card {
    background-color: #FBF9F4 !important;
    border: 1px solid #D7CCBB !important;
    border-radius: 20px !important;
    padding: 28px !important;
    margin-top: 20px !important;
}

.shap-title {
    color: #493C32 !important;
    font-family: Georgia, serif !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
}

.shap-description {
    color: #5C5147 !important;
    font-size: 16px !important;
    line-height: 1.65 !important;
}

.shap-note {
    background-color: #E6EDE1 !important;
    border-left: 5px solid #74856A !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    color: #493C32 !important;
    margin-top: 18px !important;
}

</style>
""",
unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("cardiovascular_xgboost_model.pkl")

features = joblib.load("model_features.pkl")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="section-label">CARDIOVASCULAR HEALTH ASSESSMENT</div>',
    unsafe_allow_html=True
)

st.title("CardioRisk")

st.write(
    "A machine-learning based cardiovascular risk assessment "
    "prototype designed to organize key patient health information."
)

st.divider()


# =========================================================
# 01 PERSONAL PROFILE
# =========================================================

st.markdown(
    '<div class="section-label">01 · PERSONAL PROFILE</div>',
    unsafe_allow_html=True
)

st.header("Patient Information")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=50,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
        index=0
    )

    height = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=220.0,
        value=170.0,
        step=0.5
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        value=70.0,
        step=0.5
    )


with col2:

    cholesterol = st.selectbox(
        "Cholesterol Level",
        [1, 2, 3],
        index=0,
        format_func=lambda x: {
            1: "Normal",
            2: "Above Normal",
            3: "Well Above Normal"
        }[x]
    )

    gluc = st.selectbox(
        "Glucose Level",
        [1, 2, 3],
        index=0,
        format_func=lambda x: {
            1: "Normal",
            2: "Above Normal",
            3: "Well Above Normal"
        }[x]
    )

    ap_hi = st.number_input(
        "Systolic Blood Pressure (mmHg)",
        min_value=70,
        max_value=250,
        value=120,
        step=1
    )

    ap_lo = st.number_input(
        "Diastolic Blood Pressure (mmHg)",
        min_value=40,
        max_value=150,
        value=80,
        step=1
    )


# =========================================================
# 02 LIFESTYLE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-label">02 · LIFESTYLE</div>',
    unsafe_allow_html=True
)

st.header("Daily Habits")

life1, life2, life3 = st.columns(3)


with life1:

    smoke = st.selectbox(
        "Smoking",
        [0, 1],
        index=0,
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


with life2:

    alco = st.selectbox(
        "Alcohol Consumption",
        [0, 1],
        index=0,
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


with life3:

    active = st.selectbox(
        "Physical Activity",
        [0, 1],
        index=0,
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


diet = st.selectbox(
    "Diet Quality",
    [
        "Healthy",
        "Moderately Healthy",
        "Needs Improvement"
    ],
    index=0
)


# =========================================================
# 03 MEDICAL HISTORY
# =========================================================

st.divider()

st.markdown(
    '<div class="section-label">03 · MEDICAL HISTORY</div>',
    unsafe_allow_html=True
)

st.header("Health Conditions")

med1, med2 = st.columns(2)


with med1:

    diabetes = st.selectbox(
        "Diabetes",
        ["No", "Yes"]
    )

    previous_cvd = st.selectbox(
        "Previous Cardiovascular Disease",
        ["No", "Yes"]
    )

    family_history = st.selectbox(
        "Family History of Heart Disease",
        ["No", "Yes"]
    )


with med2:

    hypertension = st.selectbox(
        "Diagnosed Hypertension",
        ["No", "Yes"]
    )

    kidney_disease = st.selectbox(
        "Kidney Disease",
        ["No", "Yes"]
    )

    other_condition = st.selectbox(
        "Other Major Health Condition",
        [
            "None",
            "Thyroid Condition",
            "Respiratory Condition",
            "Liver Condition",
            "Other"
        ]
    )


# =========================================================
# 04 ADDITIONAL HEALTH
# =========================================================

st.divider()

st.markdown(
    '<div class="section-label">04 · ADDITIONAL HEALTH</div>',
    unsafe_allow_html=True
)

st.header("Health & Wellbeing")

health1, health2, health3 = st.columns(3)


with health1:

    medication = st.selectbox(
        "Currently Taking Medication",
        ["No", "Yes"]
    )


with health2:

    stress_level = st.selectbox(
        "Stress Level",
        [
            "Low",
            "Moderate",
            "High"
        ]
    )


with health3:

    sleep_quality = st.selectbox(
        "Sleep Quality",
        [
            "Good",
            "Average",
            "Poor"
        ]
    )


# =========================================================
# FEATURE ENGINEERING
# =========================================================

gender_value = 1 if gender == "Female" else 2

bmi = weight / ((height / 100) ** 2)

pulse_pressure = ap_hi - ap_lo

high_bp = int(
    (ap_hi >= 140) or (ap_lo >= 90)
)


# =========================================================
# 05 HEALTH METRICS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-label">05 · HEALTH METRICS</div>',
    unsafe_allow_html=True
)

st.header("Health Snapshot")


# IMPORTANT:
# These are native Streamlit metrics.
# No HTML is used here.

m1, m2, m3 = st.columns(3)


with m1:

    st.metric(
        label="BODY MASS INDEX",
        value=f"{bmi:.2f}"
    )


with m2:

    st.metric(
        label="PULSE PRESSURE",
        value=f"{pulse_pressure} mmHg"
    )


with m3:

    st.metric(
        label="BLOOD PRESSURE",
        value=f"{ap_hi}/{ap_lo}"
    )


# =========================================================
# BLOOD PRESSURE INTERPRETATION
# =========================================================

if ap_hi < 120 and ap_lo < 80:

    bp_status = "Within the normal range"

elif ap_hi < 130 and ap_lo < 80:

    bp_status = "Elevated systolic pressure"

elif ap_hi >= 140 or ap_lo >= 90:

    bp_status = "High blood pressure range"

else:

    bp_status = "Above the optimal range"


st.info(
    f"Blood Pressure: {bp_status}. "
    f"Recorded reading: {ap_hi}/{ap_lo} mmHg."
)


# =========================================================
# MODEL INPUT
# =========================================================

patient_data = {

    "age_years": age,

    "gender": gender_value,

    "height": height,

    "weight": weight,

    "bmi": bmi,

    "ap_hi": ap_hi,

    "ap_lo": ap_lo,

    "pulse_pressure": pulse_pressure,

    "cholesterol": cholesterol,

    "gluc": gluc,

    "smoke": smoke,

    "alco": alco,

    "active": active,

    "high_bp": high_bp
}


patient_df = pd.DataFrame(
    [patient_data]
)


patient_df = patient_df[features]


# =========================================================
# 06 ASSESSMENT
# =========================================================

st.divider()

st.markdown(
    '<div class="section-label">06 · ASSESSMENT</div>',
    unsafe_allow_html=True
)

st.header("Cardiovascular Risk Assessment")


predict = st.button(
    "Assess Cardiovascular Risk",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    probability = model.predict_proba(
        patient_df
    )[0][1]


    prediction = int(
        probability >= 0.5
    )


    # =====================================================
    # RISK CATEGORY
    # =====================================================

    if probability < 0.30:

        risk = "Low Risk"

    elif probability < 0.60:

        risk = "Moderate Risk"

    else:

        risk = "High Risk"


    # =====================================================
    # 07 RISK ASSESSMENT
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-label">07 · RESULT</div>',
        unsafe_allow_html=True
    )

    st.header("Risk Assessment")


    # IMPORTANT:
    # Native Streamlit metrics.
    # No HTML here.

    r1, r2, r3 = st.columns(3)


    with r1:

        st.metric(
            label="ESTIMATED PROBABILITY",
            value=f"{probability * 100:.2f}%"
        )


    with r2:

        st.metric(
            label="RISK CATEGORY",
            value=risk
        )


    with r3:

        st.metric(
            label="BMI",
            value=f"{bmi:.2f}"
        )


    # =====================================================
    # MODEL MESSAGE
    # =====================================================

    if prediction == 1:

        st.warning(
            "The model estimates a higher likelihood of "
            "cardiovascular disease based on the model inputs."
        )

    else:

        st.success(
            "The model estimates a lower likelihood of "
            "cardiovascular disease based on the model inputs."
        )


    # =====================================================
    # 08 BLOOD PRESSURE ANALYSIS
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-label">08 · BLOOD PRESSURE</div>',
        unsafe_allow_html=True
    )

    st.header("Blood Pressure Analysis")


    # IMPORTANT:
    # Native Streamlit metrics.
    # No HTML here.

    bp1, bp2 = st.columns(2)


    with bp1:

        st.metric(
            label="SYSTOLIC PRESSURE",
            value=f"{ap_hi} mmHg"
        )


    with bp2:

        st.metric(
            label="DIASTOLIC PRESSURE",
            value=f"{ap_lo} mmHg"
        )


    # =====================================================
    # 09 PATIENT SUMMARY
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-label">09 · PATIENT SUMMARY</div>',
        unsafe_allow_html=True
    )

    st.header("Patient Health Profile")


    summary = {

        "Parameter": [

            "Age",

            "Gender",

            "Height",

            "Weight",

            "BMI",

            "Systolic Blood Pressure",

            "Diastolic Blood Pressure",

            "Cholesterol",

            "Glucose",

            "Smoking",

            "Alcohol Consumption",

            "Physical Activity",

            "Diet Quality",

            "Diabetes",

            "Previous Cardiovascular Disease",

            "Family History of Heart Disease",

            "Diagnosed Hypertension",

            "Kidney Disease",

            "Other Major Health Condition",

            "Current Medication",

            "Stress Level",

            "Sleep Quality"

        ],


        "Value": [

            f"{age} years",

            gender,

            f"{height:.1f} cm",

            f"{weight:.1f} kg",

            f"{bmi:.2f}",

            f"{ap_hi} mmHg",

            f"{ap_lo} mmHg",

            {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[cholesterol],

            {
                1: "Normal",
                2: "Above Normal",
                3: "Well Above Normal"
            }[gluc],

            "Yes" if smoke else "No",

            "Yes" if alco else "No",

            "Yes" if active else "No",

            diet,

            diabetes,

            previous_cvd,

            family_history,

            hypertension,

            kidney_disease,

            other_condition,

            medication,

            stress_level,

            sleep_quality

        ]

    }


    st.dataframe(
        pd.DataFrame(summary),
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # 10 MODEL EXPLANATION
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-label">10 · MODEL EXPLANATION</div>',
        unsafe_allow_html=True
    )

    st.header("SHAP Feature Explanation")


    # -----------------------------------------------------
    # SHAP DESCRIPTION
    # -----------------------------------------------------

    st.subheader(
        "Why did the model make this prediction?"
    )

    st.write(
        "SHAP values show how the patient's input features "
        "influenced the machine-learning prediction. "
        "Larger absolute values indicate stronger influence."
    )


    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    if shap is None:

        st.warning(
            "SHAP is not installed."
        )

        st.code(
            "pip install shap"
        )

    else:

        try:

            explainer = shap.TreeExplainer(
                model
            )


            shap_values = explainer.shap_values(
                patient_df
            )


            # Handle different SHAP output formats

            if isinstance(
                shap_values,
                list
            ):

                if len(shap_values) > 1:

                    local_values = shap_values[1][0]

                else:

                    local_values = shap_values[0][0]

            else:

                local_values = shap_values

                if hasattr(
                    local_values,
                    "ndim"
                ):

                    if local_values.ndim == 2:

                        if local_values.shape[1] == len(features):

                            local_values = local_values[0]

                        elif local_values.shape[0] == len(features):

                            local_values = local_values[:, 0]


            shap_df = pd.DataFrame(
                {
                    "Feature": list(features),
                    "SHAP Value": list(local_values)
                }
            )


            shap_df["Absolute Impact"] = (
                shap_df["SHAP Value"].abs()
            )


            shap_df = shap_df.sort_values(
                "Absolute Impact",
                ascending=False
            ).head(10)


            # -------------------------------------------------
            # FEATURE NAME DISPLAY
            # -------------------------------------------------

            feature_names = {

                "age_years": "Age",

                "gender": "Gender",

                "height": "Height",

                "weight": "Weight",

                "bmi": "BMI",

                "ap_hi": "Systolic BP",

                "ap_lo": "Diastolic BP",

                "pulse_pressure": "Pulse Pressure",

                "cholesterol": "Cholesterol",

                "gluc": "Glucose",

                "smoke": "Smoking",

                "alco": "Alcohol",

                "active": "Physical Activity",

                "high_bp": "High BP"

            }


            shap_df["Display Feature"] = (
                shap_df["Feature"]
                .map(feature_names)
                .fillna(shap_df["Feature"])
            )


            st.subheader(
                "Most Influential Features"
            )


            chart_df = shap_df[
                ["Display Feature", "SHAP Value"]
            ].copy()


            chart_df = chart_df.set_index(
                "Display Feature"
            )


            st.bar_chart(
                chart_df,
                use_container_width=True
            )


            # -------------------------------------------------
            # SHAP TABLE
            # -------------------------------------------------

            st.subheader(
                "Feature Influence"
            )


            display_shap = shap_df[
                [
                    "Display Feature",
                    "SHAP Value"
                ]
            ].copy()


            display_shap.columns = [
                "Feature",
                "SHAP Value"
            ]


            display_shap["Direction"] = (
                display_shap["SHAP Value"]
                .apply(
                    lambda value:
                    "Increases Risk"
                    if value > 0
                    else "Decreases Risk"
                )
            )


            st.dataframe(
                display_shap,
                use_container_width=True,
                hide_index=True
            )


            st.info(
                "Positive SHAP values push the prediction "
                "toward the positive cardiovascular-risk class. "
                "Negative values push the prediction away from "
                "that class. Larger absolute values indicate "
                "stronger influence."
            )


        except Exception as shap_error:

            st.warning(
                "SHAP explanation could not be generated "
                "for this model."
            )

            st.caption(
                f"Technical detail: {shap_error}"
            )


    # =====================================================
    # 11 MODEL INFORMATION
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-label">11 · MODEL INFORMATION</div>',
        unsafe_allow_html=True
    )

    st.header(
        "How the Assessment Works"
    )


    st.info(
        "The cardiovascular risk probability is generated "
        "by the trained machine-learning model using the "
        "clinical and lifestyle variables included during "
        "model training.\n\n"
        "Additional health-history information collected "
        "in this assessment is included in the patient "
        "profile for documentation and future model development."
    )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.divider()

    st.info(
        "This application is an educational and research "
        "prototype. The prediction is not a medical diagnosis "
        "and should not replace evaluation by a qualified "
        "healthcare professional."
    )