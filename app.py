import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# 1. Page Configuration (The "Premium" Setup)
st.set_page_config(
    page_title="EduPredict Pro | Helly Shah",
    page_icon="🎓",
    layout="wide"
)

# 2. Mock Data Generation (Simulating the Student Dataset)
@st.cache_data
def get_data():
    np.random.seed(42)
    hours = np.random.randint(1, 15, 100)
    attendance = np.random.randint(60, 100, 100)
    # Simple linear relationship with some noise
    marks = (hours * 4) + (attendance * 0.4) + np.random.randint(1, 10, 100)
    return pd.DataFrame({'Hours': hours, 'Attendance': attendance, 'Marks': marks})

df = get_data()

# 3. Sidebar Navigation
st.sidebar.title("📊 Control Panel")
menu = st.sidebar.radio("Navigate", ["Dashboard", "Machine Learning"])

# --- PAGE 1: DASHBOARD ---
if menu == "Dashboard":
    st.title("🎓 Student Performance Analytics")
    st.markdown("---")
    
    # Top Row Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Avg. Study Hours", f"{df['Hours'].mean():.1f}h")
    col3.metric("Avg. Marks", f"{df['Marks'].mean():.1f}%")

    st.markdown("### Visual Data Exploration")
    
    # Interactive Plotly Chart (Much better than static charts)
    fig = px.scatter(df, x="Hours", y="Marks", size="Attendance", 
                     color="Marks", title="Hours vs Marks (Size by Attendance)",
                     template="plotly_dark") # Dark theme for the chart
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: MACHINE LEARNING ---
else:
    st.title("🔮 AI Prediction System")
    st.info("Adjust the factors in the sidebar to see real-time predictions.")

    # ML Logic
    X = df[['Hours', 'Attendance']]
    y = df['Marks']
    model = LinearRegression().fit(X, y)

    # The logic you asked about (placed in the sidebar)
    st.sidebar.subheader("Adjust Factors")
    h_input = st.sidebar.slider("Daily Study Hours", 0, 15, 5)
    a_input = st.sidebar.slider("Attendance Percentage", 0, 100, 75)

    # Automatic Prediction
    prediction = model.predict([[h_input, a_input]])
    
    # Display Result in a "Premium" card style
    st.markdown("### 🏆 Prediction Result")
    st.success(f"### Predicted Final Score: **{prediction[0]:.2f}%**")
    
    # Small UI breakdown
    st.write(f"Based on **{h_input} hours** of study and **{a_input}% attendance**, the AI model forecasts this score.")

    st.markdown("---")
    st.markdown("#### Model Details")
    st.text(f"Algorithm: Linear Regression\nAccuracy Score: {model.score(X, y):.4f}")
    
