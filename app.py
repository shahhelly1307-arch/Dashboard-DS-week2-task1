import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. PROFESSIONAL CONFIG
st.set_page_config(page_title="EduPredict Pro", layout="wide")

# 2. PREMIUM CSS (Obsidian & Blue Aura)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .prediction-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #111119 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-text { color: #00ffcc; font-size: 42px; font-weight: bold; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 3. DATA LOADING (Updated to match dataset structure)
@st.cache_data
def load_data():
    file_name = "student_data.csv"
    if os.path.exists(file_name):
        return pd.read_csv(file_name)
    # Default data matching the screenshots' structure
    data = {
        'gender': ['female', 'female', 'female', 'male', 'male'] * 200,
        'race/ethnicity': ['group B', 'group C', 'group B', 'group A', 'group C'] * 200,
        'parental level of education': ["bachelor's degree", "some college", "master's degree", "associate's degree", "some college"] * 200,
        'lunch': ['standard', 'standard', 'standard', 'free/reduced', 'standard'] * 200,
        'test preparation course': ['none', 'completed', 'none', 'none', 'none'] * 200,
        'math score': [72, 69, 90, 47, 76] * 200,
        'reading score': [72, 90, 95, 57, 78] * 200,
        'writing score': [74, 88, 93, 44, 75] * 200
    }
    return pd.DataFrame(data)

df = load_data()

# 4. HEADER (Reference: d5cdcc6a-90cd-4f19-bbef-ab34ce7d4c94)
st.title("🎓 Student Performance Dashboard")
st.markdown("Data Analysis + Machine Learning Prediction System")
st.write("---")

# 5. DATASET PREVIEW (Reference: d5cdcc6a-90cd-4f19-bbef-ab34ce7d4c94)
st.subheader("📂 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# 6. KEY METRICS (Reference: 659c9406-3b0a-4551-b87d-154b43e36562)
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Students", len(df))
with col2:
    st.metric("Average Math Score", round(df['math score'].mean(), 2))
with col3:
    st.metric("Average Reading Score", round(df['reading score'].mean(), 2))

# 7. FILTERED VIEW (Reference: 659c9406-3b0a-4551-b87d-154b43e36562)
st.subheader("🔍 Filtered Data (Female)")
female_df = df[df['gender'] == 'female']
st.dataframe(female_df.head(), use_container_width=True)

st.write("---")

# 8. VISUALIZATIONS
st.subheader("📈 Visualizations")

# Math Score Distribution (Reference: d60c17ce-985e-4b50-b448-826132af5cc4)
st.write("### Math Score Distribution")
fig_dist = px.histogram(df, x="math score", nbins=30, marginal="box", 
                         title="Distribution of Math Scores", 
                         color_discrete_sequence=['#00ffcc'],
                         template="plotly_dark")
st.plotly_chart(fig_dist, use_container_width=True)

# Gender Performance (Reference: 7623975f-1efd-4a7d-a1d7-700d6861bb90)
st.write("### Average Math Scores by Gender")
gender_avg = df.groupby('gender')['math score'].mean().reset_index()
fig_gender = px.bar(gender_avg, x='gender', y='math score', 
                    color='gender', 
                    template="plotly_dark",
                    color_discrete_map={'female': '#00ffcc', 'male': '#1f77b4'})
st.plotly_chart(fig_gender, use_container_width=True)

# Reading vs Writing Scatter (Reference: 6f4c66a9-a371-4a09-86b9-7e90137cc657)
st.write("### Reading Score vs Writing Score")
fig_scatter = px.scatter(df, x="reading score", y="writing score", 
                         color="gender", 
                         template="plotly_dark",
                         hover_data=['math score'])
st.plotly_chart(fig_scatter, use_container_width=True)

# Correlation Heatmap (Reference: 37e8452d-bc2b-4630-bc6f-dcd003b94aac)
st.write("### Correlation Heatmap")
numeric_df = df[['math score', 'reading score', 'writing score']]
corr = numeric_df.corr()
fig_heat = px.imshow(corr, text_auto=True, 
                      aspect="auto", 
                      color_continuous_scale='RdBu_r',
                      template="plotly_dark")
st.plotly_chart(fig_heat, use_container_width=True)

# 9. AI PREDICTION SECTION
st.write("---")
st.subheader("🛡️ EduPredict AI Forecast")

col_inp, col_res = st.columns([1, 1])

with col_inp:
    reading_inp = st.slider("Reading Proficiency", 0, 100, 70)
    writing_inp = st.slider("Writing Proficiency", 0, 100, 70)
    
    # ML Logic
    X = df[['reading score', 'writing score']]
    y = df['math score']
    model = LinearRegression().fit(X, y)
    prediction = model.predict([[reading_inp, writing_inp]])[0]
    prediction = max(0, min(100, prediction))

with col_res:
    st.markdown(f"""
        <div class="prediction-card">
            <p style="color: #8b949e; text-transform: uppercase; letter-spacing: 2px;">Math Competency Forecast</p>
            <div class="metric-text">{prediction:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        gauge = {
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': "#00ffcc"},
            'steps': [
                {'range': [0, 40], 'color': "rgba(255, 118, 117, 0.2)"},
                {'range': [40, 75], 'color': "rgba(255, 234, 167, 0.2)"},
                {'range': [75, 100], 'color': "rgba(85, 239, 196, 0.2)"}
            ],
        }
    ))
    fig_gauge.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

# 10. PROFESSIONAL FOOTER
st.markdown("---")
st.caption("Lead Developer: Helly Shah | EduPredict Pro AI System")
