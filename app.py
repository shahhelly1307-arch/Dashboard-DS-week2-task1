import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression

# 1. PAGE CONFIG
st.set_page_config(page_title="EduPredict Pro", layout="wide")

# 2. STYLING
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

# 3. DATA LOADING
@st.cache_data
def load_data():
    file_name = "student_data.csv"
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        # Fallback if file is missing
        df = pd.DataFrame({
            'Hours_Studied': np.random.randint(1, 10, 100),
            'Attendance_Percentage': np.random.randint(60, 100, 100),
            'Previous_Score': np.random.randint(40, 100, 100),
            'Final_Score': np.random.randint(40, 100, 100)
        })
    
    # Cleaning column names to remove any hidden spaces
    df.columns = df.columns.str.strip()
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
cols = st.columns(3)
cols[0].metric("Total Students", len(df))

# Using the column names visible in your heatmap (44847613-3500-4ddf-b667-cb0759ce21c1)
if 'Final_Score' in df.columns:
    cols[1].metric("Avg Final Score", f"{df['Final_Score'].mean():.1f}")
if 'Attendance_Percentage' in df.columns:
    cols[2].metric("Avg Attendance", f"{df['Attendance_Percentage'].mean():.1f}%")

st.write("---")

# 7. VISUALIZATIONS
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Score Distribution")
    score_col = 'Final_Score' if 'Final_Score' in df.columns else df.columns[1]
    fig1 = px.histogram(df, x=score_col, nbins=15, template="plotly_dark", color_discrete_sequence=['#00ffcc'])
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("🔗 Score Correlation")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        fig2 = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

# 8. PREDICTION ENGINE (Mapped to your specific columns)
st.write("---")
st.subheader("🛡️ EduPredict AI Forecast")

# Check for the columns seen in your screenshot heatmap
required = ['Hours_Studied', 'Attendance_Percentage', 'Previous_Score', 'Final_Score']
if all(k in df.columns for k in required):
    c1, c2 = st.columns([1, 1])
    with c1:
        hours = st.slider("Hours Studied", 1, 12, 5)
        attendance = st.slider("Attendance %", 0, 100, 85)
        prev_score = st.slider("Previous Score", 0, 100, 70)
        
        # ML Logic
        X = df[['Hours_Studied', 'Attendance_Percentage', 'Previous_Score']]
        y = df['Final_Score']
        model = LinearRegression().fit(X, y)
        prediction = model.predict([[hours, attendance, prev_score]])[0]
        prediction = max(0, min(100, prediction))
    
    with c2:
        st.markdown(f"""
            <div class="prediction-card">
                <p style="color: #8b949e; text-transform: uppercase; font-size: 14px;">Math Competency Prediction</p>
                <div class="metric-text">{prediction:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("Prediction tool requires 'Hours_Studied', 'Attendance_Percentage', and 'Previous_Score' columns.")

# 9. FOOTER
st.markdown("---")
st.caption("Lead Developer: Helly Shah | System: EduPredict Pro AI")
