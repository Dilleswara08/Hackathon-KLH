import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(page_title="Smart Kitchen Waste Monitor", layout="wide")
st.title("🍽 Smart Kitchen Waste Monitor")
st.caption("AI-Powered Food Waste Prediction Dashboard")

# ======================================
# FILE UPLOAD SECTION
# ======================================
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Food Waste CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload your dataset to continue.")
    st.stop()

# ======================================
# LOAD DATA
# ======================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

data = load_data(uploaded_file)

# ======================================
# DATA PREPROCESSING
# ======================================
try:
    data.rename(columns={
        "Date": "date",
        "Prepared": "quantity",
        "Waste": "food_waste",
        "Temperature": "temperature"
    }, inplace=True)

    data["date"] = pd.to_datetime(data["date"])

    if "Event" in data.columns:
        data["event"] = data["Event"].apply(
            lambda x: 1 if str(x).lower() == "yes" else 0
        )
    else:
        data["event"] = 0

    data.fillna(data.mean(numeric_only=True), inplace=True)

except Exception as e:
    st.error("Dataset format incorrect. Please check column names.")
    st.stop()

st.success("Dataset Uploaded Successfully ✅")

# ======================================
# MODEL TRAINING
# ======================================
features = ["quantity", "temperature", "event"]
X = data[features]
y = data["food_waste"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = r2_score(y_test, model.predict(X_test))

st.subheader("📈 Model Performance")
st.metric("R2 Score", round(accuracy, 2))

# ======================================
# KPI SECTION
# ======================================
st.header("📊 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Quantity Prepared", int(data["quantity"].sum()))
col2.metric("Total Waste", round(data["food_waste"].sum(), 2))
col3.metric("Average Waste per Day", round(data["food_waste"].mean(), 2))

# ======================================
# PREDICTION SECTION
# ======================================
st.header("🔮 Waste Prediction")

quantity = st.slider(
    "Expected Quantity (Prepared Food)",
    int(data["quantity"].min()),
    int(data["quantity"].max()),
    int(data["quantity"].mean())
)

temperature = st.slider(
    "Temperature",
    int(data["temperature"].min()),
    int(data["temperature"].max()),
    int(data["temperature"].mean())
)

event = st.selectbox("Is There an Event?", [0, 1])

input_data = pd.DataFrame(
    [[quantity, temperature, event]],
    columns=features
)

predicted_waste = model.predict(input_data)[0]
recommended_quantity = quantity - predicted_waste

st.subheader("Prediction Result")
st.write("Predicted Waste:", round(predicted_waste, 2))
st.write("Recommended Final Quantity:", round(recommended_quantity, 2))

# ======================================
# REPORT SECTION
# ======================================
st.header("📅 Reports")

report_type = st.selectbox(
    "Select Report Type",
    ["Daily", "Monthly", "Yearly"]
)

if report_type == "Daily":
    daily = data.groupby(data["date"].dt.date)["food_waste"].sum()
    st.line_chart(daily)

elif report_type == "Monthly":
    monthly = data.groupby(data["date"].dt.month)["food_waste"].sum()
    st.bar_chart(monthly)

elif report_type == "Yearly":
    yearly = data.groupby(data["date"].dt.year)["food_waste"].sum()
    st.bar_chart(yearly)

# ======================================
# VISUAL ANALYSIS (SMALL SIZE)
# ======================================
st.header("📉 Quantity vs Waste Analysis")

fig, ax = plt.subplots(figsize=(4, 3))  # smaller chart
ax.scatter(data["quantity"], data["food_waste"])
ax.set_xlabel("Quantity Prepared")
ax.set_ylabel("Food Waste")
ax.set_title("Quantity vs Waste")
st.pyplot(fig)

# ======================================
# WASTE REDUCTION IMPACT
# ======================================
st.header("💰 Waste Reduction Impact")

baseline = 0.15 * quantity
reduction = baseline - predicted_waste

st.write("Baseline Waste (15% Rule):", round(baseline, 2))
st.write("AI Predicted Waste:", round(predicted_waste, 2))
st.write("Estimated Reduction:", round(reduction, 2))

st.success("🎉 AI-powered system helps reduce waste and improve profitability!")