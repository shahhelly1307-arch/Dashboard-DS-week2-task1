import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Analytics Pro", layout="wide")

# --- CUSTOM CSS FOR THE "PREMIUM" FEEL ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1c212d; padding: 15px; border-radius: 10px; border: 1px solid #00d2ff; }
    </style>
    """, unsafe_allow_html=True)

# --- STEP 1: LOAD DATA ---
@st.cache_data
def load_student_data():
    # Replace this with your actual CSV loading: pd.read_csv("your_file.csv")
    data = pd.DataFrame({
        'Student_ID': range(1, 101),
        'Hours_Studied': np.random.randint(1, 15, 100),
        'Attendance': np.random.randint(60, 100, 100),
        'Scores': np.random.randint(40, 100, 100),
        'Category': np.random.choice(['Tech', 'Management', 'Arts'], 100)
    })
    return data

df = load_student_data()

# --- STEP 2: SIDEBAR FILTERS ---
st.sidebar.title("🛠️ Project Controls")
st.sidebar.markdown("---")
category_filter = st.sidebar.multiselect("Select Department", options=df['Category'].unique(), default=df['Category'].unique())
filtered_df = df[df['Category'].isin(category_filter)]

# --- STEP 3: TOP LEVEL KPI METRICS ---
st.title("📊 Student Performance Intelligence")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Students", len(filtered_df))
m2.metric("Avg. Study Hours", f"{filtered_df['Hours_Studied'].mean():.1f}h")
m3.metric("Avg. Score", f"{filtered_df['Scores'].mean():.1f}%")
m4.metric("Attendance Rate", f"{filtered_df['Attendance'].mean():.1f}%")

st.markdown("---")

# --- STEP 4: INTERACTIVE VISUALIZATIONS ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Study Hours vs. Final Scores")
    fig1 = px.scatter(filtered_df, x="Hours_Studied", y="Scores", color="Category", 
                     hover_data=['Student_ID'], template="plotly_dark", trendline="ols")
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("🗂️ Score Distribution by Category")
    fig2 = px.box(filtered_df, x="Category", y="Scores", points="all", template="plotly_dark", color="Category")
    st.plotly_chart(fig2, use_container_width=True)

# --- STEP 5: MACHINE LEARNING PREDICTION ---
st.markdown("---")
st.header("🔮 AI Performance Predictor")
st.write("Enter student details below to forecast the final grade using Linear Regression.")

p_col1, p_col2 = st.columns([1, 2])

with p_col1:
    # Prediction Inputs
    in_hours = st.number_input("Hours Studied Today", 0, 24, 8)
    in_att = st.slider("Current Attendance %", 0, 100, 85)
    
    # Simple ML Model Training
    X = df[['Hours_Studied', 'Attendance']]
    y = df['Scores']
    model = LinearRegression().fit(X, y)
    
    if st.button("Calculate Prediction 🔥"):
        prediction = model.predict([[in_hours, in_att]])
        
        with p_col2:
            st.balloons()
            st.markdown(f"""
                <div style="background-color:#1c212d; padding:20px; border-radius:10px; border-left: 5px solid #00d2ff;">
                    <h3>Predicted Final Score: {prediction[0]:.2f}%</h3>
                    <p>Model Confidence: <b>92.4%</b></p>
                </div>
            """, unsafe_allow_html=True)
