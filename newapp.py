import streamlit as st
import joblib
import pandas as pd

# Daftar fitur sesuai urutan model
FEATURES = [
    'Age',
    'Chest Pain',
    'Persistent Cough',
    'Snoring/Sleep Apnea',
    'Excessive Sweating',
    'Cold Hands/Feet',
    'Shortness of Breath',
    'Chest Discomfort (Activity)',
    'Pain in Neck/Jaw/Shoulder/Back',
    'Nausea/Vomiting',
    'Fatigue & Weakness',
    'Anxiety/Feeling of Doom',
    'Swelling (Edema)',
    'High Blood Pressure',
    'Irregular Heartbeat',
    'Dizziness'
]

# Load model pipeline
model = joblib.load("svm_stroke_model.pkl")

# Custom CSS for styling
st.markdown("""
<style>
body { font-family: Arial, sans-serif; background: #111; color: #e74c3c; }
.stApp { background: #111; color: #e74c3c; }
.container { max-width: 500px; margin: 0 auto; }
.card { background: #222; padding: 30px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.1);}
h2 { text-align: center; color: #e74c3c; }
.form-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
.form-group { margin-bottom: 10px; }
label { display: block; margin-bottom: 5px; color: #e74c3c; }
select, input { width: 100%; padding: 8px; border-radius: 5px; border: 1px solid #ccc; color: #e74c3c; background: #111; }
button, .stButton>button { background: #222; color: #e74c3c; padding: 10px; border: none; border-radius: 8px; width: 100%; margin-top: 10px; font-weight: bold; }
.result { margin-top: 20px; padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; color: #e74c3c; background: #222; }
.risk { background: #222; color: #e74c3c; }
.safe { background: #222; color: #e74c3c; }
.error { background: #222; color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="container"><h2>Stroke Risk Prediction</h2><div class="card">', unsafe_allow_html=True)

with st.form("stroke_form"):
    # Grid layout for form fields
    cols = st.columns(2)
    input_data = {}
    for idx, feature in enumerate(FEATURES):
        with cols[idx % 2]:
            st.markdown(f'<div class="form-group">', unsafe_allow_html=True)
            st.markdown(f'<label>{feature}:</label>', unsafe_allow_html=True)
            if feature == "Age":
                input_data[feature] = st.number_input(" ", min_value=1, max_value=120, step=1, key=feature)
            else:
                input_data[feature] = st.selectbox(
                    " ", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key=feature
                )[1]
            st.markdown('</div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("Predict Risk")

prediction = None
if submitted:
    try:
        df = pd.DataFrame([input_data])[FEATURES]
        result = model.predict(df)[0]
        prediction = "At Risk" if result == 1 else "Not at Risk"
        css_class = "risk" if prediction == "At Risk" else "safe"
    except Exception as e:
        prediction = f"Error: {str(e)}"
        css_class = "error"
    st.markdown(
        f'<div class="result {css_class}">{prediction}</div>',
        unsafe_allow_html=True
    )

st.markdown('</div></div>', unsafe_allow_html=True)
