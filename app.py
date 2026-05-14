import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression

# 1. CONFIG
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

# 3. ROBUST DATA LOADING
@st.cache_data
def load_data():
    file_name = "student_data.csv"
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        # Emergency backup data if file is missing
        df = pd.DataFrame({
            'gender': ['female', 'male'] * 25,
            'math_score': np.random.randint(40, 100, 50),
            'reading_score': np.random.randint(40, 100, 50),
            'writing_score': np.random.randint(40, 100, 50)
        })
    
    # DYNAMIC COLUMN MAPPING: Fixes the KeyError from 3c48f001-6950-4d69-9669-3035b9d04422
    # This searches for columns containing specific words to avoid strict name errors
    cols = df.columns.tolist()
    mapping = {}
    for c in cols:
        low_c = c.lower()
        if 'math' in low_c: mapping['math'] = c
        if 'read' in low_c: mapping['reading'] = c
        if 'writ' in low_c: mapping['writing'] = c
        if 'gend' in low_c: mapping['gender'] = c
    
    # Rename columns to standard internal names
    df = df.rename(columns={v: k for k, v in mapping.items()})
    return df

try:
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
    
    # Safe check for columns before calculating
    if 'math' in df.columns:
        m2.metric("Average Math Score", f"{df['math'].mean():.2f}")
    if 'reading' in df.columns:
        m3.metric("Average Reading Score", f"{df['reading'].mean():.2f}")

    st.write("---")

    # 7. VISUALIZATIONS
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("### Math Score Distribution")
        if 'math' in df.columns:
            fig_dist = px.histogram(df, x="math", nbins=20, template="plotly_dark", color_discrete_sequence=['#00ffcc'])
            st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.write("### Performance by Gender")
        if 'gender' in df.columns and 'math' in df.columns:
            fig_gender = px.bar(df.groupby('gender')['math'].mean().reset_index(), 
                                x='gender', y='math', template="plotly_dark")
            st.plotly_chart(fig_gender, use_container_width=True)

    # Correlation Heatmap
    st.write("### Score Correlation")
    numeric_cols = [c for c in ['math', 'reading', 'writing'] if c in df.columns]
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)

    # 8. AI PREDICTION
    st.write("---")
    st.subheader("🛡️ EduPredict AI Forecast")
    
    if 'math' in df.columns and 'reading' in df.columns and 'writing' in df.columns:
        r_val = st.slider("Reading Input", 0, 100, 70)
        w_val = st.slider("Writing Input", 0, 100, 70)

        model = LinearRegression().fit(df[['reading', 'writing']], df['math'])
        pred = max(0, min(100, model.predict([[r_val, w_val]])[0]))

        st.markdown(f"""
            <div class="prediction-card">
                <p style="color: #8b949e; text-transform: uppercase;">Forecasted Math Score</p>
                <div class="metric-text">{pred:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Cannot run prediction: Required score columns are missing from the CSV.")

except Exception as e:
    st.error(f"Critical System Error: {e}")
    st.info("Check if your CSV file column names contain 'math', 'reading', and 'writing'.")

# 9. FOOTER
st.caption("Lead Developer: Helly Shah")
