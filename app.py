import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 1. Page Config
st.set_page_config(page_title="Student Analytics Pro", layout="wide")

# 2. Load Data
@st.cache_data
def load_data():
    # Make sure 'student_data.csv' is uploaded to your GitHub
    return pd.read_csv("student_data.csv")

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.title("🛠️ Dashboard Controls")
category = st.sidebar.multiselect("Filter by ID Range", options=df['Student_ID'].unique(), default=df['Student_ID'].unique()[:10])
filtered_df = df[df['Student_ID'].isin(category)]

# --- SECTION 1: KEY METRICS (KPIs) ---
st.title("🎓 Student Performance Intelligence")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Students", len(df))
m2.metric("Avg. Study Hours", f"{df['Hours_Studied'].mean():.1f}h")
m3.metric("Avg. Attendance", f"{df['Attendance_Percentage'].mean():.1f}%")
m4.metric("Mean Final Score", f"{df['Final_Score'].mean():.1f}")

st.divider()

# --- SECTION 2: DATA VISUALIZATION (EDA) ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Hours vs Final Score")
    fig1 = px.scatter(df, x="Hours_Studied", y="Final_Score", 
                     color="Final_Score", size="Attendance_Percentage",
                     template="plotly_dark", trendline="ols")
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("📊 Score Distribution")
    fig2 = px.histogram(df, x="Final_Score", nbins=15, 
                        marginal="box", template="plotly_dark",
                        color_discrete_sequence=['#00d2ff'])
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- SECTION 3: THE PREDICTION SYSTEM (As per your Image) ---
st.header("🔮 AI Prediction Engine")

# ML Model Training
X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
y = df['Final_Score']
model = LinearRegression().fit(X, y)

# Prediction Sliders
st.subheader("Adjust Parameters")
p_col1, p_col2 = st.columns(2)
with p_col1:
    h_in = st.slider("Reading Score Input (Hours Studied)", 0, 15, 8)
with p_col2:
    a_in = st.slider("Writing Score Input (Attendance %)", 0, 100, 85)
    p_in = 70 # Hidden or set as default to keep UI simple

# Prediction Calculation
prediction = model.predict([[h_in, a_in, p_in]])[0]
prediction = max(0, min(100, prediction)) # Clamp between 0-100

# THE GREEN PREDICTION BOX
st.markdown(f"""
    <div style="background-color: #e8f5e9; padding: 25px; border-radius: 15px; border-left: 10px solid #2e7d32; margin-top: 20px;">
        <h1 style="color: #2e7d32; margin: 0; font-family: sans-serif; font-size: 32px;">
            Predicted Math Competency: {prediction:.2f}%
        </h1>
    </div>
    """, unsafe_allow_html=True)

# THE GAUGE CHART
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prediction,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Confidence Level", 'font': {'size': 24}},
    gauge = {
        'axis': {'range': [0, 100]},
        'bar': {'color': "#2e7d32"},
        'steps': [
            {'range': [0, 40], 'color': "#ff8a80"},
            {'range': [40, 75], 'color': "#fff59d"},
            {'range': [75, 100], 'color': "#69f0ae"}
        ],
    }
))
fig_gauge.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "gray"})
st.plotly_chart(fig_gauge, use_container_width=True)

# FOOTER
st.sidebar.divider()
st.sidebar.info("Developed by Helly Shah\nNational Research Award Winner")
