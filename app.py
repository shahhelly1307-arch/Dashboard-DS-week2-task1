import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression

# 1. PAGE CONFIGURATION - Ensures proper scaling on mobile
st.set_page_config(page_title="EduPredict AI", layout="centered")

# 2. DATA LOADING
@st.cache_data
def load_data():
    file_name = "student_data.csv"
    if not os.path.exists(file_name):
        data = {
            'Hours_Studied': [4, 9, 2, 10, 5],
            'Attendance_Percentage': [85, 95, 70, 98, 80],
            'Previous_Score': [70, 88, 50, 92, 65],
            'Final_Score': [72, 90, 55, 95, 68],
            'Sleep_Hours': [7, 6, 8, 7, 6]
        }
        return pd.DataFrame(data)
    return pd.read_csv(file_name)

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛠️ Navigation")
# I moved "AI Risk Predictor" to the TOP so it opens first
menu = st.sidebar.radio("Go to:", ["AI Risk Predictor", "Executive Dashboard", "Business Case Study"])

# --- PAGE 1: AI RISK PREDICTOR (Now the Default Start Page) ---
if menu == "AI Risk Predictor":
    st.title("🔮 AI Risk Predictor")
    st.markdown("### Adjust Student Inputs")

    # Train Model
    X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
    y = df['Final_Score']
    model = LinearRegression().fit(X, y)

    # INPUT SLIDERS
    reading_in = st.slider("Reading Score Input", 0, 100, 57)
    writing_in = st.slider("Writing Score Input", 0, 100, 81)
    
    # Calculate Prediction
    prediction = model.predict([[reading_in/10, writing_in, 70]])[0]
    prediction = max(0, min(100, prediction))

    # SCROLLABLE GREEN BOX
    st.markdown(f"""
        <div style="background-color: #eafaf1; padding: 25px; border-radius: 15px; border-left: 10px solid #2e7d32; margin-bottom: 20px;">
            <h2 style="color: #1e8449; font-family: sans-serif; font-size: 28px;">
                Predicted Math Competency: {prediction:.2f}%
            </h2>
        </div>
        """, unsafe_allow_html=True)

    # SCROLLABLE GAUGE
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2e7d32", 'thickness': 0.2},
            'steps': [
                {'range': [0, 35], 'color': '#ff7675'},
                {'range': [35, 70], 'color': '#ffeaa7'},
                {'range': [70, 100], 'color': '#55efc4'}
            ],
        }
    ))
    
    fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- PAGE 2: EXECUTIVE DASHBOARD ---
elif menu == "Executive Dashboard":
    st.title("🎓 Executive Dashboard")
    st.write("Full analytics view of student data.")
    
    # use_container_width=True makes this scrollable!
    fig_scatter = px.scatter(df, x="Hours_Studied", y="Final_Score", color="Attendance_Percentage")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- PAGE 3: CASE STUDY ---
else:
    st.title("📂 Business Case Study")
    st.write("Project focus: Student retention and risk analysis.")

# FOOTER
st.sidebar.markdown("---")
st.sidebar.write("**Developer:** Helly Shah")
