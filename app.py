import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression

# 1. PROFESSIONAL CONFIG
st.set_page_config(page_title="EduPredict Pro", layout="centered")

# 2. PREMIUM CSS (Obsidian & Blue Aura)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .prediction-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #111119 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-text { color: #00ffcc; font-size: 42px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 3. DATA LOADING (Uses your updated CSV)
@st.cache_data
def load_data():
    file_name = "student_data.csv"
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    return pd.DataFrame({'Hours_Studied':[5,10], 'Attendance_Percentage':[80,95], 'Previous_Score':[60,90], 'Final_Score':[65,92], 'Sleep_Hours':[7,6]})

df = load_data()

# 4. HEADER
st.title("🛡️ EduPredict AI Pro")
st.markdown("### Professional Student Analytics Dashboard")
st.write("---")

# 5. INPUT SECTION
st.subheader("📊 Assessment Parameters")
reading = st.slider("Reading Proficiency", 0, 100, 57)
writing = st.slider("Writing Proficiency", 0, 100, 81)
attendance = st.slider("Attendance %", 0, 100, 92)

# 6. AI PREDICTION LOGIC
X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
y = df['Final_Score']
model = LinearRegression().fit(X, y)
prediction = model.predict([[reading/10, attendance, writing]])[0]
prediction = max(0, min(100, prediction))

# 7. DISPLAY PREDICTION CARD
st.markdown(f"""
    <div class="prediction-card">
        <p style="color: #8b949e; text-transform: uppercase; letter-spacing: 2px;">Math Competency Forecast</p>
        <div class="metric-text">{prediction:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# 8. DISPLAY GAUGE CHART
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    gauge = {
        'axis': {'range': [0, 100], 'tickcolor': "white"},
        'bar': {'color': "#00ffcc"},
        'steps': [
            {'range': [0, 40], 'color': "rgba(255, 118, 117, 0.2)"},
            {'range': [40, 75], 'color': "rgba(255, 234, 167, 0.2)"},
            {'range': [75, 100], 'color': "rgba(85, 239, 196, 0.2)"}
        ],
    }
))
fig_gauge.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, margin=dict(l=20,r=20,t=30,b=20))
st.plotly_chart(fig_gauge, use_container_width=True)

# 9. DISPLAY ANALYSIS GRAPH (The part that was missing!)
st.write("---")
st.subheader("📈 Trend Analysis: Hours vs Score")
fig_scatter = px.scatter(df, x="Hours_Studied", y="Final_Score", 
                         trendline="ols", 
                         color="Attendance_Percentage",
                         template="plotly_dark")
st.plotly_chart(fig_scatter, use_container_width=True)

# 10. PROFESSIONAL FOOTER
st.markdown("---")
st.caption("Lead Developer: Helly Shah | National Research Award Winner")
