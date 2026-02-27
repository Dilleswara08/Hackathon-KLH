import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime

# ===============================
# CONFIGURATION
# ===============================
st.set_page_config(page_title="Smart Kitchen Waste Monitor", layout="wide")
st.title("🍽 Smart Kitchen Waste Monitor")
st.caption("AI-Powered Waste Prediction with Live Weather Integration")

API_KEY = "d38a07a0aaee472b6bfab29627bbb06d"

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Configuration")
city = st.sidebar.text_input("Enter City", "Hyderabad")
uploaded_file = st.sidebar.file_uploader("Upload Restaurant CSV (Optional)", type=["csv"])

# ===============================
# WEATHER API FUNCTION
# ===============================
def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        temp = data["main"]["temp"]
        rain = 1 if "rain" in data else 0
        return temp, rain
    except:
        return None, None

temperature_live, rain_live = get_weather(city)

# ===============================
# DATA GENERATION / INGESTION
# ===============================
def generate_data():
    np.random.seed(42)
    days = 365
    df = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=days),
        "sales": np.random.randint(200, 500, days),
        "temperature": np.random.randint(20, 40, days),
        "rain": np.random.randint(0, 2, days),
        "event": np.random.randint(0, 2, days)
    })
    df["food_waste"] = (
        0.1 * df["sales"] +
        5 * df["rain"] -
        2 * df["event"] +
        np.random.normal(0, 5, days)
    ).abs()
    return df

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data["date"] = pd.to_datetime(data["date"])
    st.success("Custom dataset loaded successfully.")
else:
    data = generate_data()
    st.info("Using synthetic demo dataset.")

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# ===============================
# MODEL TRAINING
# ===============================
features = ["sales", "temperature", "rain", "event"]
target = "food_waste"

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = r2_score(y_test, model.predict(X_test))

st.subheader("📈 Model Performance")
st.metric("R2 Score", round(accuracy, 2))

# ===============================
# LIVE PREDICTION
# ===============================
st.header("🔮 Live Waste Prediction & Order Optimization")

col1, col2 = st.columns(2)

with col1:
    sales = st.slider("Expected Sales Today", 200, 600, 400)
    event = st.selectbox("Local Event Today?", [0, 1])

with col2:
    if temperature_live is not None:
        st.write(f"🌡 Live Temperature in {city}: {temperature_live}°C")
        temperature = temperature_live
    else:
        temperature = st.slider("Temperature (°C)", 20, 45, 30)

    rain = rain_live if rain_live is not None else st.selectbox("Rain?", [0, 1])

input_df = pd.DataFrame([[sales, temperature, rain, event]], columns=features)
predicted_waste = model.predict(input_df)[0]

recommended_order = sales - predicted_waste

if event == 1:
    recommended_order *= 1.10
if rain == 1:
    recommended_order *= 0.95

st.subheader("📊 Prediction Results")
st.write("Predicted Waste:", round(predicted_waste, 2))
st.write("Recommended Order Quantity:", round(recommended_order, 2))

# ===============================
# REPORTS SECTION
# ===============================
st.header("📅 Reports & Analytics")

report_type = st.selectbox("Select Report Type", ["Daily", "Weekly", "Yearly"])

if report_type == "Daily":
    daily = data.groupby(data["date"].dt.date)["food_waste"].sum()
    st.line_chart(daily)

elif report_type == "Weekly":
    weekly = data.groupby(data["date"].dt.isocalendar().week)["food_waste"].sum()
    st.bar_chart(weekly)

elif report_type == "Yearly":
    yearly = data.groupby(data["date"].dt.year)["food_waste"].sum()
    st.bar_chart(yearly)

# ===============================
# VISUAL ANALYSIS
# ===============================
st.header("📉 Sales vs Waste Analysis")

fig, ax = plt.subplots()
ax.scatter(data["sales"], data["food_waste"])
ax.set_xlabel("Sales")
ax.set_ylabel("Food Waste")
ax.set_title("Sales vs Waste Relationship")
st.pyplot(fig)

# ===============================
# IMPACT CALCULATION
# ===============================
st.header("💰 Waste Reduction Impact")

baseline_waste = 0.15 * sales
reduction = baseline_waste - predicted_waste

st.write("Baseline Waste (15% assumption):", round(baseline_waste, 2))
st.write("AI Predicted Waste:", round(predicted_waste, 2))
st.write("Estimated Reduction:", round(reduction, 2))

st.success("🎉 AI-powered system helps restaurants reduce waste and increase profitability.")