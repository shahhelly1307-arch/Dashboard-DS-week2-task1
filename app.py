import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression

# 1. PROFESSIONAL CONFIG
st.set_page_config(page_title="EduPredict Pro", layout="wide")

# 2. PREMIUM CSS
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
    </style>
    """, unsafe_allow_html=True)

# 3. DATA LOADING & CLEANING
@st.cache_data
def load_data():
    file_name = "student_data.csv"
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        # Fallback dummy data matching the screenshots
        data = {
            'gender': ['female', 'female', 'female', 'male', 'male'] * 10,
            'math_score': [72, 69, 90, 47, 76] * 10,
            'reading_score': [72, 90, 95, 57, 78] * 10,
            'writing_score': [74, 88, 93, 44, 75] * 10
        }
        df = pd.DataFrame(data)
    
    # CLEANING: This prevents the KeyError from
    # It converts 'Math Score' or 'math score' to 'math_score'
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df

df = load_data()

# 4. HEADER
st.title("🎓 Student Performance Dashboard")
st.markdown("Data Analysis + Machine Learning Prediction System")
st.write("---")

# 5. DATASET PREVIEW
st.subheader("📂 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# 6. KEY METRICS
st.subheader("📊 Key Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("Total Students", len(df))
# Using underscores to match the cleaning step
m2.metric("Average Math Score", f"{df['math_score'].mean():.2f}")
m3.metric("Average Reading Score", f"{df['reading_score'].mean():.2f}")

st.write("---")

# 7. VISUALIZATIONS
col_a, col_b = st.columns(2)

with col_a:
    # Math Score Distribution
    st.write("### Math Score Distribution")
    fig_dist = px.histogram(df, x="math_score", nbins=20, 
                             template="plotly_dark", color_discrete_sequence=['#00ffcc'])
    st.plotly_chart(fig_dist, use_container_width=True)

with col_b:
    # Average Math Scores by Gender
    st.write("### Average Math Scores by Gender")
    fig_gender = px.bar(df.groupby('gender')['math_score'].mean().reset_index(), 
                        x='gender', y='math_score', template="plotly_dark")
    st.plotly_chart(fig_gender, use_container_width=True)

# Reading vs Writing Scatter
st.write("### Reading Score vs Writing Score")
fig_scatter = px.scatter(df, x="reading_score", y="writing_score", color="gender", template="plotly_dark")
st.plotly_chart(fig_scatter, use_container_width=True)

# Correlation Heatmap
st.write("### Correlation Heatmap")
corr = df[['math_score', 'reading_score', 'writing_score']].corr()
fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark")
st.plotly_chart(fig_heat, use_container_width=True)

# 8. AI PREDICTION SECTION
st.write("---")
st.subheader("🛡️ EduPredict AI Forecast")
read_val = st.slider("Reading Score", 0, 100, 70)
write_val = st.slider("Writing Score", 0, 100, 70)

# ML Logic
X = df[['reading_score', 'writing_score']]
y = df['math_score']
model = LinearRegression().fit(X, y)
pred = max(0, min(100, model.predict([[read_val, write_val]])[0]))

st.markdown(f"""
    <div class="prediction-card">
        <p style="color: #8b949e; text-transform: uppercase;">Predicted Math Score</p>
        <div class="metric-text">{pred:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# 9. FOOTER
st.caption("Lead Developer: Helly Shah")

