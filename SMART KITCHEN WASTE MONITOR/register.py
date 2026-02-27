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

# ======================================
# SESSION STATE INIT
# ======================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_data" not in st.session_state:
    st.session_state.user_data = {}

if "dataset" not in st.session_state:
    st.session_state.dataset = None

# ======================================
# LOGIN / REGISTRATION PAGE
# ======================================
if not st.session_state.logged_in:

    st.title("🔐 Smart Kitchen Waste Monitor Login")
    st.subheader("Register Your Facility")

    name = st.text_input("Owner Name")
    restaurant = st.text_input("Restaurant Name")
    address = st.text_area("Restaurant Address")
    unique_id = st.text_input("Create Unique ID")
    password = st.text_input("Create Password", type="password")
    uploaded_file = st.file_uploader("Upload Food Waste Dataset (CSV)", type=["csv"])

    if st.button("Register & Login"):

        if not all([name, restaurant, address, unique_id, password, uploaded_file]):
            st.warning("Please fill all fields and upload dataset.")
        else:
            st.session_state.logged_in = True
            st.session_state.user_data = {
                "name": name,
                "restaurant": restaurant,
                "address": address,
                "unique_id": unique_id
            }
            st.session_state.dataset = pd.read_csv(uploaded_file)
            st.success("Registration Successful! Redirecting to Dashboard...")
            st.rerun()

# ======================================
# DASHBOARD
# ======================================
else:

    data = st.session_state.dataset

    st.title("🍽 Smart Kitchen Waste Monitor Dashboard")
    st.caption("AI-Powered Food Waste Prediction System")

    # Logout Button
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.dataset = None
        st.rerun()

    # Show User Info
    st.sidebar.header("🏢 Facility Info")
    st.sidebar.write("Owner:", st.session_state.user_data["name"])
    st.sidebar.write("Restaurant:", st.session_state.user_data["restaurant"])
    st.sidebar.write("Address:", st.session_state.user_data["address"])
    st.sidebar.write("Unique ID:", st.session_state.user_data["unique_id"])

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

    except Exception:
        st.error("Dataset format incorrect.")
        st.stop()

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

    event_option = st.selectbox("Is There an Event?", ["No", "Yes"])
    event = 1 if event_option == "Yes" else 0

    input_data = pd.DataFrame(
        [[quantity, temperature, event]],
        columns=features
    )

    predicted_waste = model.predict(input_data)[0]
    recommended_quantity = quantity - predicted_waste

    colA, colB = st.columns(2)
    colA.metric("Predicted Waste", round(predicted_waste, 2))
    colB.metric("Recommended Final Quantity", round(recommended_quantity, 2))

    # ======================================
    # REPORTS
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
    # VISUAL ANALYSIS
    # ======================================
    st.header("📉 Quantity vs Waste Analysis")

    fig, ax = plt.subplots(figsize=(4, 3))
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