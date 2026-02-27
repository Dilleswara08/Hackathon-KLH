import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Smart Kitchen Waste Monitor", layout="wide")

st.title("🍽 Smart Kitchen Waste Monitor Dashboard")

# -----------------------------
# KEY OBJECTIVES SECTION
# -----------------------------
st.header("🎯 Key Objectives")
st.markdown("""
- ✅ Reduce restaurant food waste  
- ✅ Optimize perishable item ordering  
- ✅ Correlate external factors with waste  
- ✅ Improve restaurant profitability  
""")

# -----------------------------
# REQUIREMENTS SECTION
# -----------------------------
st.header("📌 System Requirements")
st.markdown("""
- ✔ Sales and weather data ingestion  
- ✔ Regression model for waste prediction  
- ✔ Order quantity recommendation engine  
- ✔ Local events data integration  
""")

# -----------------------------
# DELIVERABLES SECTION
# -----------------------------
st.header("📦 Deliverables")
st.markdown("""
1. Waste prediction agent  
2. Order optimization dashboard  
3. Demo with sample restaurant data  
4. Waste reduction impact report  
""")

st.divider()

# -----------------------------
# DATA GENERATION
# -----------------------------
np.random.seed(42)
days = 365

data = pd.DataFrame({
    "sales": np.random.randint(200, 500, days),
    "temperature": np.random.randint(20, 40, days),
    "rain": np.random.randint(0, 2, days),
    "event": np.random.randint(0, 2, days)
})

data["food_waste"] = (
    0.1 * data["sales"]
    + 5 * data["rain"]
    - 2 * data["event"]
    + np.random.normal(0, 5, days)
).abs()

# -----------------------------
# MODEL TRAINING
# -----------------------------
X = data[["sales", "temperature", "rain", "event"]]
y = data["food_waste"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

st.success(f"📊 Model Accuracy (R2 Score): {round(accuracy,2)}")

# -----------------------------
# USER INPUT SECTION
# -----------------------------
st.header("🔮 Predict Waste & Recommend Order")

col1, col2 = st.columns(2)

with col1:
    sales = st.slider("Expected Sales", 200, 600, 400)
    temperature = st.slider("Temperature (°C)", 20, 45, 30)

with col2:
    rain = st.selectbox("Rain?", [0,1])
    event = st.selectbox("Local Event?", [0,1])

# Prediction
input_data = np.array([[sales, temperature, rain, event]])
predicted_waste = model.predict(input_data)[0]

# Recommendation Logic
recommended_order = sales - predicted_waste
if event == 1:
    recommended_order *= 1.10
if rain == 1:
    recommended_order *= 0.95

st.subheader("📈 Prediction Results")
st.write("Predicted Waste:", round(predicted_waste,2))
st.write("Recommended Order Quantity:", round(recommended_order,2))

# -----------------------------
# VISUALIZATION
# -----------------------------
st.header("📊 Waste vs Sales Visualization")

fig, ax = plt.subplots()
ax.scatter(data["sales"], data["food_waste"])
ax.set_xlabel("Sales")
ax.set_ylabel("Food Waste")
ax.set_title("Sales vs Waste Relationship")

st.pyplot(fig)

# -----------------------------
# IMPACT REPORT
# -----------------------------
st.header("📉 Waste Reduction Impact")

baseline_waste = 0.15 * sales
improvement = baseline_waste - predicted_waste

st.write("Baseline Waste (15% assumption):", round(baseline_waste,2))
st.write("Estimated Waste After Model:", round(predicted_waste,2))
st.write("Estimated Reduction:", round(improvement,2))

st.success("🎉 This AI-driven system helps restaurants reduce waste and increase profit.")