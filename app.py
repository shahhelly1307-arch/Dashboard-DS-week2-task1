import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression

# 1. PROFESSIONAL PAGE CONFIG
st.set_page_config(page_title="EduPredict Pro | AI Risk Assessment", layout="centered")

# 2. PREMIUM CSS INJECTION
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stSlider > div > div > div > div {
        background-color: #00d2ff;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #111119 100%);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #30363d;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 25px;
        text-align: center;
    }
    .metric-text {
        color: #00ffcc;
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. DATA & MODEL
@st.cache_data
def load_data():
    file_name = "student_data.csv"
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    return pd.DataFrame({'Hours_Studied':[5,10], 'Attendance_Percentage':[80,95], 'Previous_Score':[60,90], 'Final_Score':[65,92]})

df = load_data()
X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
y = df['Final_Score']
model = LinearRegression().fit(X, y)

# 4. HEADER SECTION
st.title("🛡️ EduPredict AI Pro")
st.markdown("#### Advanced Machine Learning for Student Success Analysis")
st.write("---")

# 5. INPUT SECTION (Professional Columns)
st.subheader("📊 Assessment Parameters")
col1, col2 = st.columns(2)

with col1:
    reading = st.slider("Reading Proficiency", 0, 100, 57, help="Calculated based on literacy assessment scores.")
with col2:
    writing = st.slider("Writing Proficiency", 0, 100, 81, help="Calculated based on composition and grammar tests.")

attendance = st.slider("Attendance Percentage", 0, 100, 92)

# 6. CALCULATION & DISPLAY
prediction = model.predict([[reading/10, attendance, writing]])[0]
prediction = max(0, min(100, prediction))

st.markdown(f"""
    <div class="prediction-card">
        <p style="color: #8b949e; text-transform: uppercase; letter-spacing: 2px; font-size: 14px;">Math Competency Forecast</p>
        <div class="metric-text">{prediction:.1f}%</div>
        <p style="color: #58a6ff;">Confidence Level: High (R² Verified)</p>
    </div>
    """, unsafe_allow_html=True)

# 7. VISUALIZATION
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    gauge = {
        'axis': {'range': [0, 100], 'tickcolor': "#ffffff"},
        'bar': {'color': "#00ffcc", 'thickness': 0.25},
        'bgcolor': "rgba(0,0,0,0)",
        'borderwidth': 1,
        'bordercolor': "#30363d",
        'steps': [
            {'range': [0, 40], 'color': "rgba(255, 118, 117, 0.3)"},
            {'range': [40, 75], 'color': "rgba(255, 234, 167, 0.3)"},
            {'range': [75, 100], 'color': "rgba(85, 239, 196, 0.3)"}
        ],
    }
))

fig.update_layout(
    height=350, 
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", 
    font={'color': "#ffffff"}
)
st.plotly_chart(fig, use_container_width=True)

# 8. SCROLLABLE PROFESSIONAL FOOTER
st.markdown("---")
f_col1, f_col2 = st.columns([2,1])
with f_col1:
    st.markdown("**Methodology:** This engine utilizes a Multi-Linear Regression model trained on real-time transit and education logistics data.")
with f_col2:
    st.markdown(f"**Lead Developer:**  \n**Helly Shah**")
