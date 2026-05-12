import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LogisticRegression, LinearRegression

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="EduPredict AI | Case Study Edition", layout="wide")

# 2. SMART DATA LOADING (With Burnout & Risk Logic)
@st.cache_data
def load_and_process_data():
    file_name = "student_data.csv"
    
    if not os.path.exists(file_name):
        st.error(f"❌ **File Not Found:** I cannot see '{file_name}' in your GitHub.")
        st.stop()
        
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.strip()
    
    # Feature Engineering for Case Study
    if 'Sleep_Hours' in df.columns:
        df['Burnout_Index'] = df['Hours_Studied'] / (df['Sleep_Hours'] + 0.1)
    else:
        df['Burnout_Index'] = df['Hours_Studied'] / 8 # Fallback logic
        
    df['At_Risk'] = (df['Final_Score'] < 55).astype(int)
    return df

df = load_and_process_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛠️ Decision Support")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", ["Executive Dashboard", "AI Risk Predictor", "Business Case Study"])

# --- PAGE 1: EXECUTIVE DASHBOARD ---
if menu == "Executive Dashboard":
    st.title("🎓 Student Performance Analytics")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students", len(df))
    m2.metric("Avg. Marks", f"{df['Final_Score'].mean():.1f}%")
    m3.metric("Burnout Alert", f"{df[df['Burnout_Index'] > 2.5].shape[0]} Students")
    m4.metric("Retention Rate", f"{(1 - df['At_Risk'].mean())*100:.1f}%")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Performance vs. Effort")
        fig1 = px.scatter(df, x="Hours_Studied", y="Final_Score", color="Burnout_Index",
                         size="Attendance_Percentage", template="plotly_dark", 
                         color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig1, use_container_width=True)
    
    with c2:
        st.subheader("📊 Score Distribution")
        fig2 = px.histogram(df, x="Final_Score", nbins=15, template="plotly_dark", 
                            color_discrete_sequence=['#00d2ff'])
        st.plotly_chart(fig2, use_container_width=True)

# --- PAGE 2: AI RISK PREDICTOR (Matches your Screenshots) ---
elif menu == "AI Risk Predictor":
    st.title("🔮 Predictive Intelligence")
    
    # Train Prediction Model
    X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
    y = df['Final_Score']
    reg_model = LinearRegression().fit(X, y)
    
    st.markdown("#### Adjust Student Inputs")
    p_col1, p_col2 = st.columns(2)
    with p_col1: h_in = st.slider("Reading Score Input (Hours)", 0, 15, 8)
    with p_col2: a_in = st.slider("Writing Score Input (Attendance)", 0, 100, 75)
    prev_in = 70 # Default background value

    # Prediction
    res = reg_model.predict([[h_in, a_in, prev_in]])[0]
    res = max(0, min(100, res))

    # GREEN PREDICTION BOX
    st.markdown(f"""
        <div style="background-color: #e8f5e9; padding: 25px; border-radius: 15px; border-left: 10px solid #2e7d32; margin: 20px 0;">
            <h1 style="color: #2e7d32; margin: 0; font-family: sans-serif; font-size: 32px;">
                Predicted Math Competency: {res:.2f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)

    # EXACT MATCH GAUGE CHART
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = res,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level", 'font': {'size': 24, 'color': '#7f8c8d'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#2e7d32", 'thickness': 0.25},
            'steps': [
                {'range': [0, 40], 'color': '#ff7675'},   # Red
                {'range': [40, 70], 'color': '#ffeaa7'},  # Yellow
                {'range': [70, 100], 'color': '#55efc4'}  # Green
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'value': res}
        }
    ))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': "Arial"})
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- PAGE 3: BUSINESS CASE STUDY ---
else:
    st.title("📂 End-to-End Case Study")
    
    # Feature Importance Logic
    X_logic = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
    y_logic = df['At_Risk']
    clf = LogisticRegression().fit(X_logic, y_logic)
    importance = np.abs(clf.coef_[0])
    
    feat_df = pd.DataFrame({'Feature': X_logic.columns, 'Importance': importance}).sort_values(by='Importance')
    
    st.subheader("🎯 Feature Importance (Business Drivers)")
    fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', template="plotly_dark")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    ### 📑 Business Recommendations
    * **Attendance Alerts:** Set threshold at 75% for intervention.
    * **Burnout Workshops:** Target students with Index > 2.5.
    * **Resource Planning:** Re-allocate mentors to Red Zone students.
    """)

# FOOTER
st.sidebar.markdown("---")
st.sidebar.write("**Developer:** Helly Shah")
st.sidebar.caption("National Research Award Winner")
