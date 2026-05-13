import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression

# 1. MOBILE-OPTIMIZED CONFIG
st.set_page_config(page_title="EduPredict Pro", layout="centered")

# 2. DATA & MODEL
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

# 3. COMPACT PROFESSIONAL STYLING
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .prediction-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #111119 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-text { color: #00ffcc; font-size: 36px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 4. HEADER
st.title("🛡️ EduPredict AI Pro")
st.write("Adjust sliders and scroll down to view the analysis graph.")

# 5. COMPACT INPUTS
reading = st.slider("Reading Proficiency", 0, 100, 57)
writing = st.slider("Writing Proficiency", 0, 100, 81)
attendance = st.slider("Attendance %", 0, 100, 92)

# 6. PREDICTION
prediction = model.predict([[reading/10, attendance, writing]])[0]
prediction = max(0, min(100, prediction))

st.markdown(f"""
    <div class="prediction-card">
        <p style="color: #8b949e; text-transform: uppercase; font-size: 12px;">Math Competency Forecast</p>
        <div class="metric-text">{prediction:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# 7. THE GRAPH (Ensuring Visibility)
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    gauge = {
        'axis': {'range': [0, 100], 'tickcolor': "#ffffff"},
        'bar': {'color': "#00ffcc", 'thickness': 0.2},
        'steps': [
            {'range': [0, 40], 'color': "rgba(255, 118, 117, 0.3)"},
            {'range': [40, 75], 'color': "rgba(255, 234, 167, 0.3)"},
            {'range': [75, 100], 'color': "rgba(85, 239, 196, 0.3)"}
        ],
    }
))

fig.update_layout(
    height=300, # Smaller height makes it easier to see everything
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor="rgba(0,0,0,0)", 
    font={'color': "#ffffff"}
)

st.plotly_chart(fig, use_container_width=True)

# 8. FOOTER
st.markdown("---")
st.caption("Developed by Helly Shah")
