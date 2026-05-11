import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(page_title="Student Performance AI", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

df = load_data()

# --- SIDEBAR ---
st.sidebar.title("Settings")
st.sidebar.info("Upload 'student_data.csv' to your GitHub for this to work.")

# --- HEADER SECTION ---
st.title("🎓 Student Performance Dashboard")
st.markdown("Analyze patterns and predict success using Machine Learning.")

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", len(df))
col2.metric("Avg. Study Hours", f"{df['Hours_Studied'].mean():.1f}h")
col3.metric("Avg. Attendance", f"{df['Attendance_Percentage'].mean():.1f}%")
col4.metric("Mean Final Score", f"{df['Final_Score'].mean():.1f}")

st.divider()

# --- VISUALIZATIONS ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("📊 Hours vs Final Score")
    fig = px.scatter(df, x="Hours_Studied", y="Final_Score", 
                     color="Final_Score", size="Attendance_Percentage",
                     template="plotly_dark", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("📈 Score Distribution")
    fig2 = px.histogram(df, x="Final_Score", nbins=20, 
                        marginal="box", template="plotly_dark",
                        color_discrete_sequence=['#00d2ff'])
    st.plotly_chart(fig2, use_container_width=True)

# --- PREDICTION SYSTEM (The Highlight) ---
st.divider()
st.header("🔮 Marks Prediction System")
st.write("Adjust parameters in the sidebar to see the predicted score.")

# Model Training
X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
y = df['Final_Score']
model = LinearRegression().fit(X, y)

# Sidebar Inputs for Prediction
st.sidebar.divider()
st.sidebar.subheader("Predict Marks")
h_in = st.sidebar.slider("Study Hours", 1, 15, 8)
a_in = st.sidebar.slider("Attendance %", 50, 100, 85)
p_in = st.sidebar.slider("Previous Score", 30, 100, 70)

# Calculation
pred_score = model.predict([[h_in, a_in, p_in]])[0]

# Display Prediction
res_col1, res_col2 = st.columns([1, 2])
with res_col1:
    st.markdown(f"""
    <div style="background-color:#1c212d; padding:30px; border-radius:15px; border: 2px solid #00d2ff; text-align:center;">
        <h2 style="color:#00d2ff; margin:0;">{pred_score:.2f}%</h2>
        <p style="color:#fafafa; margin:0;">Predicted Final Score</p>
    </div>
    """, unsafe_allow_html=True)

with res_col2:
    # Small breakdown of the result
    st.info(f"Based on **{h_in} hours** of study and **{a_in}% attendance**, our AI model estimates a score of **{pred_score:.1f}%**.")
    st.write("This uses a Linear Regression model trained on your student dataset.")
