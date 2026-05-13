import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression

# 1. MOBILE-FIRST CONFIGURATION
# Changing to 'centered' helps the phone browser allow scrolling.
st.set_page_config(page_title="EduPredict AI", layout="centered")

# 2. DATA LOADING
@st.cache_data
def load_data():
    file_name = "student_data.csv"
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    # Fallback data
    return pd.DataFrame({'Hours_Studied': [5,10], 'Attendance_Percentage': [80,95], 'Previous_Score': [60,90], 'Final_Score': [65,92]})

df = load_data()

# 3. FORCE THE PREDICTOR TO BE THE ONLY PAGE (Removes the Sidebar Lock)
st.title("🔮 Student Risk Predictor")
st.write("Scroll down to see your results!")

# Train Model
X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
y = df['Final_Score']
model = LinearRegression().fit(X, y)

# INPUT SLIDERS - Spaced out for easy touch scrolling
st.subheader("📋 Input Details")
reading_in = st.slider("Reading Score Input", 0, 100, 57)
st.write(" ") # Adding space
writing_in = st.slider("Writing Score Input", 0, 100, 81)

# Prediction Calculation
prediction = model.predict([[reading_in/10, writing_in, 70]])[0]
prediction = max(0, min(100, prediction))

# 4. THE GREEN BOX (Optimized for Mobile Height)
st.markdown(f"""
    <div style="background-color: #eafaf1; padding: 20px; border-radius: 10px; border-left: 8px solid #2ecc71; margin: 20px 0;">
        <h2 style="color: #1e8449; font-size: 24px;">Predicted Math Competency: {prediction:.2f}%</h2>
    </div>
    """, unsafe_allow_html=True)

# 5. THE GAUGE (Small height to ensure scrolling works)
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    gauge = {
        'axis': {'range': [0, 100]},
        'bar': {'color': "#2e7d32"},
        'steps': [
            {'range': [0, 40], 'color': "#ff7675"},
            {'range': [40, 70], 'color': "#ffeaa7"},
            {'range': [70, 100], 'color': "#55efc4"}
        ],
    }
))

# Force a small height so the browser doesn't get stuck
fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# 6. FOOTER (Helps confirm scrolling works)
st.markdown("---")
st.caption("Developed by Helly Shah | National Research Award Winner")
