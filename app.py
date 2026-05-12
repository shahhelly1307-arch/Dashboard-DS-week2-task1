import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="EduPredict AI", layout="wide")

# 2. DATA LOADING
@st.cache_data
def load_data():
    file_name = "student_data.csv"
    if not os.path.exists(file_name):
        # Create dummy data if file is missing so app doesn't crash
        data = {
            'Hours_Studied': np.random.randint(1, 15, 100),
            'Attendance_Percentage': np.random.randint(60, 100, 100),
            'Previous_Score': np.random.randint(40, 100, 100),
            'Final_Score': np.random.randint(40, 100, 100),
            'Sleep_Hours': np.random.randint(5, 9, 100)
        }
        return pd.DataFrame(data)
    return pd.read_csv(file_name)

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛠️ Navigation")
menu = st.sidebar.radio("Go to:", ["Executive Dashboard", "AI Risk Predictor", "Business Case Study"])

# --- PAGE: AI RISK PREDICTOR (MATCHES YOUR IMAGES) ---
if menu == "AI Risk Predictor":
    st.title("🔮 AI Risk Predictor")
    
    # Train Model
    X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
    y = df['Final_Score']
    model = LinearRegression().fit(X, y)

    # SLIDERS (Matching "Reading Score" and "Writing Score" style from images)
    st.markdown("### Input Parameters")
    reading_in = st.slider("Reading Score Input", 0, 100, 57)
    writing_in = st.slider("Writing Score Input", 0, 100, 81)
    
    # Calculate Prediction
    # We map your sliders to the model features
    prediction = model.predict([[reading_in/10, writing_in, 70]])[0]
    prediction = max(0, min(100, prediction))

    # THE LARGE GREEN PREDICTION BOX (Matches Image)
    st.markdown(f"""
        <div style="background-color: #eafaf1; padding: 40px; border-radius: 15px; text-align: left; margin: 20px 0;">
            <h1 style="color: #1e8449; font-family: sans-serif; font-size: 42px; font-weight: bold;">
                Predicted Math Competency: {prediction:.2f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)

    # THE EXACT GAUGE (Matches Image)
    st.markdown("<h3 style='text-align: center; color: #7f8c8d;'>Confidence Level</h3>", unsafe_allow_html=True)
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
            'bar': {'color': "#2e7d32", 'thickness': 0.2},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#eeeeee",
            'steps': [
                {'range': [0, 35], 'color': '#ff7675'},   # Red section
                {'range': [35, 70], 'color': '#ffeaa7'},  # Yellow section
                {'range': [70, 100], 'color': '#55efc4'}  # Green section
            ],
        }
    ))

    fig_gauge.update_layout(
        height=450,
        margin=dict(l=30, r=30, t=0, b=0),
        paper_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "#2d3436", 'family': "Arial"}
    )

    st.plotly_chart(fig_gauge, use_container_width=True)

# --- OTHER PAGES (KEEPING YOUR DASHBOARD & CASE STUDY) ---
elif menu == "Executive Dashboard":
    st.title("🎓 Executive Dashboard")
    st.write("Full analytics view of student data.")
    st.scatter_chart(df, x="Hours_Studied", y="Final_Score")

else:
    st.title("📂 Business Case Study")
    st.markdown("### Project Goal: Identify At-Risk Students")
    st.write("Using Logistic Regression to predict dropout probability.")

# FOOTER
st.sidebar.markdown("---")
st.sidebar.write("**Developer:** Helly Shah")
