import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(page_title="Smart Kitchen Waste Monitor", layout="wide")
st.title("🍽 Smart Kitchen Waste Monitor")
st.caption("AI-Powered Waste Prediction with Live & Forecast Weather")

# 🔑 YOUR API KEY
API_KEY = "d38a07a0aaee472b6bfab29627bbb06d"

# ======================================
# SIDEBAR
# ======================================
st.sidebar.header("⚙️ Configuration")
city = st.sidebar.text_input("Enter City", "Hyderabad")
uploaded_file = st.sidebar.file_uploader("Upload Restaurant CSV (Optional)", type=["csv"])

# ======================================
# WEATHER FUNCTION
# ======================================
def get_weather_data(city):
    try:
        # Current Weather
        current_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        current_res = requests.get(current_url).json()

        if "main" not in current_res:
            return None, None, None, None

        today_temp = current_res["main"]["temp"]
        today_rain = 1 if "rain" in current_res else 0

        # Forecast Weather
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        forecast_res = requests.get(forecast_url).json()

        tomorrow_date = (datetime.now() + timedelta(days=1)).date()
        temps = []
        rain_flag = 0

        if "list" in forecast_res:
            for item in forecast_res["list"]:
                forecast_date = datetime.fromtimestamp(item["dt"]).date()
                if forecast_date == tomorrow_date:
                    temps.append(item["main"]["temp"])
                    if "rain" in item:
                        rain_flag = 1

        tomorrow_temp = sum(temps) / len(temps) if temps else None

        return today_temp, today_rain, tomorrow_temp, rain_flag

    except:
        return None, None, None, None


today_temp, today_rain, tomorrow_temp, tomorrow_rain = get_weather_data(city)

# ======================================
# DATA GENERATION
# ======================================
def generate_data():
    np.random.seed(42)
    days = 365

    df = pd.DataFrame({
        "date": pd.date_range(end=datetime.now(), periods=days),
        "sales": np.random.randint(200, 500, days),
        "temperature": np.random.randint(20, 40, days),
        "rain": np.random.randint(0, 2, days),
        "event": np.random.randint(0, 2, days)
    })

    df["food_waste"] = (
        0.1 * df["sales"]
        + 5 * df["rain"]
        - 2 * df["event"]
        + np.random.normal(0, 5, days)
    ).abs()

    return df


if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data["date"] = pd.to_datetime(data["date"])
    st.success("Custom dataset loaded successfully.")
else:
    data = generate_data()
    st.info("Using synthetic demo dataset.")

# ======================================
# MODEL TRAINING
# ======================================
features = ["sales", "temperature", "rain", "event"]

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
# WEATHER DISPLAY
# ======================================
st.header("🌦 Weather Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Today")
    st.write("Temperature:", today_temp if today_temp is not None else "N/A", "°C")
    st.write("Rain:", "Yes" if today_rain == 1 else "No")

with col2:
    st.subheader("Tomorrow")
    st.write(
        "Avg Temperature:",
        round(tomorrow_temp, 2) if tomorrow_temp is not None else "N/A",
        "°C"
    )
    st.write("Rain Expected:", "Yes" if tomorrow_rain == 1 else "No")

# ======================================
# PREDICTION SECTION
# ======================================
st.header("🔮 Waste Prediction")

sales = st.slider("Expected Sales Today", 200, 600, 400)
event = st.selectbox("Local Event Today?", [0, 1])

if today_temp is not None:
    today_input = pd.DataFrame(
        [[sales, today_temp, today_rain, event]],
        columns=features
    )

    predicted_today = model.predict(today_input)[0]
    recommended_today = sales - predicted_today

    if event == 1:
        recommended_today *= 1.10
    if today_rain == 1:
        recommended_today *= 0.95

    st.subheader("Today's Prediction")
    st.write("Predicted Waste:", round(predicted_today, 2))
    st.write("Recommended Order:", round(recommended_today, 2))

if tomorrow_temp is not None:
    tomorrow_input = pd.DataFrame(
        [[sales, tomorrow_temp, tomorrow_rain, event]],
        columns=features
    )

    predicted_tomorrow = model.predict(tomorrow_input)[0]

    st.subheader("Tomorrow Prediction")
    st.write("Predicted Waste:", round(predicted_tomorrow, 2))

# ======================================
# REPORT SECTION
# ======================================
st.header("📅 Reports")

report_type = st.selectbox(
    "Select Report Type",
    ["Daily (Current Month)", "Weekly (Current Month)", "Yearly"]
)

current_month = datetime.now().month
current_year = datetime.now().year

current_month_data = data[
    (data["date"].dt.month == current_month)
    & (data["date"].dt.year == current_year)
]

if report_type == "Daily (Current Month)":
    if not current_month_data.empty:
        daily = current_month_data.groupby(
            current_month_data["date"].dt.date
        )["food_waste"].sum()
        st.line_chart(daily)
    else:
        st.warning("No data available for current month.")

elif report_type == "Weekly (Current Month)":
    if not current_month_data.empty:
        weekly = current_month_data.groupby(
            current_month_data["date"].dt.isocalendar().week
        )["food_waste"].sum()
        st.bar_chart(weekly)
    else:
        st.warning("No data available for current month.")

elif report_type == "Yearly":
    yearly = data.groupby(
        data["date"].dt.year
    )["food_waste"].sum()
    st.bar_chart(yearly)

# ======================================
# VISUAL ANALYSIS
# ======================================
st.header("📉 Sales vs Waste Analysis")

fig, ax = plt.subplots()
ax.scatter(data["sales"], data["food_waste"])
ax.set_xlabel("Sales")
ax.set_ylabel("Food Waste")
ax.set_title("Sales vs Waste Relationship")
st.pyplot(fig)

# ======================================
# IMPACT CALCULATION
# ======================================
st.header("💰 Waste Reduction Impact")

if today_temp is not None:
    baseline = 0.15 * sales
    reduction = baseline - predicted_today

    st.write("Baseline Waste (15% rule):", round(baseline, 2))
    st.write("AI Predicted Waste:", round(predicted_today, 2))
    st.write("Estimated Reduction:", round(reduction, 2))

st.success("🎉 AI-powered system reduces waste and improves profitability.")