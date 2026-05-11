import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.title("🚀 Quick Sales Predictor")

# 1. Simple Data
data = pd.DataFrame({
    'Days': [1, 2, 3, 4, 5, 6, 7],
    'Sales': [100, 150, 130, 200, 240, 210, 300]
})

# 2. Key Metrics
st.metric("Total Sales", data['Sales'].sum())

# 3. Easy Interactive Chart
fig = px.line(data, x='Days', y='Sales', title="Sales Trend")
st.plotly_chart(fig)

# 4. Simple Prediction Logic
X = data[['Days']]
y = data['Sales']
model = LinearRegression().fit(X, y)

st.subheader("🔮 Predict Future Sales")
day_input = st.number_input("Enter Day Number", value=8)
if st.button("Predict"):
    pred = model.predict([[day_input]])
    st.success(f"Estimated Sales for Day {day_input}: ${pred[0]:.2f}")
  
