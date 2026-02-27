import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Smart Kitchen Waste Monitor - AI Pro",
    layout="wide",
    page_icon="🍽"
)

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
# LOGIN PAGE
# ======================================
if not st.session_state.logged_in:

    st.title("🔐 Smart Kitchen Waste Monitor")
    st.subheader("AI-Powered Food Waste Intelligence System")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Owner Name")
        restaurant = st.text_input("Restaurant Name")
        unique_id = st.text_input("Create Unique ID")

    with col2:
        address = st.text_area("Restaurant Address")
        password = st.text_input("Create Password", type="password")

    uploaded_file = st.file_uploader(
        "Upload Food Waste Dataset (CSV)",
        type=["csv"]
    )

    st.info("Required Columns: Date, Prepared, Waste, Temperature, Event(optional)")

    if st.button("🚀 Register & Login"):

        if not all([name, restaurant, address, unique_id, password, uploaded_file]):
            st.warning("⚠ Please fill all fields and upload dataset.")
        else:
            st.session_state.logged_in = True
            st.session_state.user_data = {
                "name": name,
                "restaurant": restaurant,
                "address": address,
                "unique_id": unique_id
            }
            st.session_state.dataset = pd.read_csv(uploaded_file)
            st.success("✅ Registration Successful!")
            st.rerun()


# ======================================
# DASHBOARD
# ======================================
else:

    if st.session_state.dataset is None:
        st.warning("Upload dataset first.")
        st.stop()

    data = st.session_state.dataset.copy()

    st.title("🍽 Smart Kitchen Waste Monitor - AI Pro Dashboard")
    st.caption("Advanced AI-Driven Waste Prediction & Business Intelligence")

    # Logout
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.dataset = None
        st.rerun()

    # Sidebar Info
    st.sidebar.header("🏢 Facility Info")
    st.sidebar.write("Owner:", st.session_state.user_data["name"])
    st.sidebar.write("Restaurant:", st.session_state.user_data["restaurant"])
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
        st.error("❌ Dataset format incorrect.")
        st.stop()

    # ======================================
    # SIDEBAR FILTERS
    # ======================================
    st.sidebar.header("📊 Filters")

    start_date = st.sidebar.date_input("Start Date", data["date"].min())
    end_date = st.sidebar.date_input("End Date", data["date"].max())

    filtered = data[
        (data["date"] >= pd.to_datetime(start_date)) &
        (data["date"] <= pd.to_datetime(end_date))
    ]

    # ======================================
    # MODEL TRAINING
    # ======================================
    features = ["quantity", "temperature", "event"]
    X = filtered[features]
    y = filtered["food_waste"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # ======================================
    # KPI SECTION
    # ======================================
    st.header("📌 Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Prepared", int(filtered["quantity"].sum()))
    col2.metric("Total Waste", round(filtered["food_waste"].sum(), 2))
    col3.metric(
        "Waste %",
        round((filtered["food_waste"].sum() /
               filtered["quantity"].sum()) * 100, 2)
    )
    col4.metric("Model Accuracy (R2)", round(r2, 2))

    st.markdown("---")

    # ======================================
    # TABS
    # ======================================
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Trend Analysis", "🔮 Prediction", "📊 Feature Insights", "📄 Reports"]
    )

    # ======================================
    # TAB 1 - TREND
    # ======================================
    with tab1:
        st.subheader("Waste Trend Over Time")

        trend = filtered.groupby(
            filtered["date"].dt.date
        )["food_waste"].sum().reset_index()

        fig = px.line(
            trend,
            x="date",
            y="food_waste",
            title="Daily Waste Trend",
            markers=True
        )

        st.plotly_chart(fig, use_container_width=True)

    # ======================================
    # TAB 2 - PREDICTION
    # ======================================
    with tab2:

        st.subheader("AI Waste Prediction")

        quantity = st.slider(
            "Expected Quantity",
            int(filtered["quantity"].min()),
            int(filtered["quantity"].max()),
            int(filtered["quantity"].mean())
        )

        temperature = st.slider(
            "Temperature",
            int(filtered["temperature"].min()),
            int(filtered["temperature"].max()),
            int(filtered["temperature"].mean())
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

        baseline = 0.15 * quantity
        reduction = baseline - predicted_waste

        st.info(f"Estimated Reduction vs 15% Rule: {round(reduction,2)}")

    # ======================================
    # TAB 3 - FEATURE IMPORTANCE
    # ======================================
    with tab3:

        st.subheader("Feature Importance Analysis")

        importance = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        })

        fig2 = px.bar(
            importance,
            x="Feature",
            y="Importance",
            color="Importance",
            title="Factors Influencing Food Waste"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ======================================
    # TAB 4 - REPORTS
    # ======================================
    with tab4:

        st.subheader("Download Report")

        csv = filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="waste_report.csv",
            mime="text/csv"
        )

    st.success("🚀 Advanced AI Dashboard Active & Investor Ready!")