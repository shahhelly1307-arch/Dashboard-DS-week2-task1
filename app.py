import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, LinearRegression

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="EduPredict AI | Case Study Edition", layout="wide")

# 2. DATA LOADING & ADVANCED FEATURE ENGINEERING
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("student_data.csv")
    # Feature Engineering: Burnout Index (High Study / Low Sleep = Risk)
    # We add a small constant to avoid division by zero
    df['Burnout_Index'] = df['Hours_Studied'] / (df['Sleep_Hours'] + 0.1)
    # Binary Label for Case Study: 1 if At-Risk (Score < 55), else 0
    df['At_Risk'] = (df['Final_Score'] < 55).astype(int)
    return df

try:
    df = load_and_process_data()
except:
    st.error("Please upload 'student_data.csv' to your GitHub repository.")
    st.stop()

# --- SIDEBAR: STRATEGIC CONTROLS ---
st.sidebar.title("🛠️ Decision Support")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", ["Executive Dashboard", "AI Risk Predictor", "Business Case Study"])

# --- PAGE 1: EXECUTIVE DASHBOARD ---
if menu == "Executive Dashboard":
    st.title("🎓 Student Performance Analytics")
    st.markdown("### Real-time Institutional Overview")
    
    # KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students", len(df), delta="Active")
    m2.metric("Avg. Marks", f"{df['Final_Score'].mean():.1f}%")
    m3.metric("Burnout Alert", f"{df[df['Burnout_Index'] > 2].shape[0]} Students", delta_color="inverse")
    m4.metric("Retention Rate", f"{(1 - df['At_Risk'].mean())*100:.1f}%")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Performance vs. Effort")
        fig1 = px.scatter(df, x="Hours_Studied", y="Final_Score", color="Burnout_Index",
                         size="Attendance_Percentage", hover_data=['Student_ID'],
                         template="plotly_dark", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        st.subheader("🔥 Burnout Analysis")
        fig2 = px.box(df, y="Burnout_Index", points="all", template="plotly_dark", color_discrete_sequence=['#ff4b4b'])
        st.plotly_chart(fig2, use_container_width=True)

# --- PAGE 2: AI RISK PREDICTOR ---
elif menu == "AI Risk Predictor":
    st.title("🔮 Predictive Intelligence")
    
    # Model Training (Internal)
    X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
    y = df['Final_Score']
    reg_model = LinearRegression().fit(X, y)
    
    st.markdown("#### Adjust Student Inputs")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1: h_in = st.slider("Daily Study Hours", 0, 15, 8)
    with p_col2: a_in = st.slider("Attendance %", 0, 100, 75)
    with p_col3: prev_in = st.slider("Previous Exam Score", 0, 100, 65)

    # Prediction
    res = reg_model.predict([[h_in, a_in, prev_in]])[0]
    res = max(0, min(100, res))

    # Visual Result (Matches Friend's style)
    st.markdown(f"""
        <div style="background-color: #e8f5e9; padding: 25px; border-radius: 15px; border-left: 10px solid #2e7d32; margin: 20px 0;">
            <h2 style="color: #2e7d32; margin: 0;">Predicted Competency: {res:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = res,
        title = {'text': "Confidence Meter"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2e7d32"},
            'steps': [
                {'range': [0, 50], 'color': "#ff8a80"},
                {'range': [50, 75], 'color': "#fff59d"},
                {'range': [75, 100], 'color': "#69f0ae"}]}))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- PAGE 3: BUSINESS CASE STUDY ---
else:
    st.title("📂 End-to-End Case Study")
    st.info("Project Goal: Identify at-risk students to reduce dropout rates.")
    
    st.markdown("""
    ### 1. Problem Statement
    High dropout rates lead to institutional loss. We aim to predict 'At-Risk' status early.
    
    ### 2. Feature Importance (Why students fail)
    """)
    
    # Logic to show which feature matters most
    X_logic = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
    y_logic = df['At_Risk']
    clf = LogisticRegression().fit(X_logic, y_logic)
    importance = np.abs(clf.coef_[0])
    
    feat_df = pd.DataFrame({'Feature': X_logic.columns, 'Importance': importance})
    fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', 
                     title="Key Factors Influencing Dropout Risk", template="plotly_dark")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    ### 3. Business Recommendations
    * **High Attendance Focus:** Attendance is the #1 predictor. Trigger alerts at < 75%.
    * **Burnout Management:** Students with Burnout Index > 2.5 require mandatory counseling.
    * **Early Intervention:** Use the AI Predictor during mid-terms to re-allocate teaching resources.
    """)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("**Developer:** Helly Shah")
st.sidebar.caption("National Research Award Winner | AI for Social Good")
