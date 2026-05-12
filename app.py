import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LogisticRegression, LinearRegression

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="EduPredict AI | Case Study Edition", layout="wide")

# 2. SMART DATA LOADING (Debug Mode Included)
@st.cache_data
def load_and_process_data():
    file_name = "student_data.csv"
    
    # Check if file exists in the GitHub root
    if not os.path.exists(file_name):
        st.error(f"❌ **File Not Found:** I cannot see '{file_name}' in your GitHub repository.")
        st.info(f"**Detected Files:** {os.listdir('.')}")
        st.stop()
        
    df = pd.read_csv(file_name)
    
    # Clean column names (removes hidden spaces or tabs)
    df.columns = df.columns.str.strip()
    
    # Required columns for the Case Study and ML logic
    required = ['Hours_Studied', 'Attendance_Percentage', 'Previous_Score', 'Sleep_Hours', 'Final_Score']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        st.error(f"❌ **Missing Columns in CSV:** {missing}")
        st.write("Current columns found in your file:", list(df.columns))
        st.stop()

    # Feature Engineering (Case Study Metrics)
    df['Burnout_Index'] = df['Hours_Studied'] / (df['Sleep_Hours'] + 0.1)
    df['At_Risk'] = (df['Final_Score'] < 55).astype(int)
    return df

# Initialize Data
df = load_and_process_data()

# --- SIDEBAR: STRATEGIC NAVIGATION ---
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
    m3.metric("Burnout Alert", f"{df[df['Burnout_Index'] > 2.5].shape[0]} Students", delta_color="inverse")
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
        st.subheader("📊 Score Distribution")
        fig2 = px.histogram(df, x="Final_Score", nbins=15, template="plotly_dark", 
                            color_discrete_sequence=['#00d2ff'], marginal="box")
        st.plotly_chart(fig2, use_container_width=True)

# --- PAGE 2: AI RISK PREDICTOR ---
elif menu == "AI Risk Predictor":
    st.title("🔮 Predictive Intelligence")
    
    # Model Training: Linear Regression for Score Prediction
    X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
    y = df['Final_Score']
    reg_model = LinearRegression().fit(X, y)
    
    st.markdown("#### Adjust Student Inputs")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1: h_in = st.slider("Daily Study Hours", 0, 15, 8)
    with p_col2: a_in = st.slider("Attendance %", 0, 100, 75)
    with p_col3: prev_in = st.slider("Previous Exam Score", 0, 100, 65)

    # Prediction Calculation
    res = reg_model.predict([[h_in, a_in, prev_in]])[0]
    res = max(0, min(100, res))

    # UI Components matching your images
    st.markdown(f"""
        <div style="background-color: #e8f5e9; padding: 25px; border-radius: 15px; border-left: 10px solid #2e7d32; margin: 20px 0;">
            <h2 style="color: #2e7d32; margin: 0; font-family: sans-serif;">Predicted Math Competency: {res:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = res,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Meter", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2e7d32"},
            'steps': [
                {'range': [0, 50], 'color': "#ff8a80"},
                {'range': [50, 75], 'color': "#fff59d"},
                {'range': [75, 100], 'color': "#69f0ae"}]}))
    
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "gray"})
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- PAGE 3: BUSINESS CASE STUDY ---
else:
    st.title("📂 End-to-End Case Study")
    st.info("**Objective:** Implement data-driven strategies to reduce dropout rates.")
    
    # Logic: Using Logistic Regression to find Feature Importance
    X_logic = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
    y_logic = df['At_Risk']
    clf = LogisticRegression().fit(X_logic, y_logic)
    importance = np.abs(clf.coef_[0])
    
    feat_df = pd.DataFrame({'Feature': X_logic.columns, 'Importance': importance}).sort_values(by='Importance', ascending=True)
    
    st.subheader("🎯 Feature Importance (The 'Why')")
    fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', 
                     template="plotly_dark", color='Importance', color_continuous_scale="Viridis")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    ### 📑 Business Recommendations
    1. **Early Warning System:** Since Attendance is a high-impact factor, trigger an automated intervention if attendance falls below **75%**.
    2. **Burnout Prevention:** Students showing a high *Burnout Index* (> 2.5) should be prioritized for wellness workshops.
    3. **Resource Allocation:** Use the *AI Predictor* during the first month to identify students needing peer mentorship.
    """)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write(f"**Developer:** Helly Shah")
st.sidebar.caption("National Research Award Winner")
