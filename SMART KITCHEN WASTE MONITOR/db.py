import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ======================================
# DATABASE SETUP
# ======================================
conn = sqlite3.connect("smart_kitchen.db", check_same_thread=False)
c = conn.cursor()

# Create Users Table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    restaurant TEXT,
    address TEXT,
    unique_id TEXT UNIQUE,
    password TEXT
)
""")

# Create Dataset Table
c.execute("""
CREATE TABLE IF NOT EXISTS waste_data (
    user_id INTEGER,
    date TEXT,
    quantity REAL,
    food_waste REAL,
    temperature REAL,
    event INTEGER
)
""")

# Create Prediction History Table
c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    user_id INTEGER,
    quantity REAL,
    temperature REAL,
    event INTEGER,
    predicted_waste REAL
)
""")

conn.commit()

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(page_title="Smart Kitchen Waste Monitor", layout="wide")

# ======================================
# SESSION STATE
# ======================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ======================================
# LOGIN / REGISTER
# ======================================
if not st.session_state.logged_in:

    st.title("🔐 Smart Kitchen Waste Monitor")

    menu = st.radio("Select Option", ["Login", "Register"])

    unique_id = st.text_input("Unique ID")
    password = st.text_input("Password", type="password")

    if menu == "Register":
        name = st.text_input("Owner Name")
        restaurant = st.text_input("Restaurant Name")
        address = st.text_area("Address")
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

        if st.button("Register"):

            if not all([name, restaurant, address, unique_id, password, uploaded_file]):
                st.warning("Please fill all fields.")
            else:
                try:
                    c.execute("INSERT INTO users (name, restaurant, address, unique_id, password) VALUES (?, ?, ?, ?, ?)",
                              (name, restaurant, address, unique_id, password))
                    conn.commit()

                    user_id = c.lastrowid

                    data = pd.read_csv(uploaded_file)
                    data.rename(columns={
                        "Date": "date",
                        "Prepared": "quantity",
                        "Waste": "food_waste",
                        "Temperature": "temperature"
                    }, inplace=True)

                    data["date"] = pd.to_datetime(data["date"])
                    data["event"] = 0

                    for _, row in data.iterrows():
                        c.execute("INSERT INTO waste_data VALUES (?, ?, ?, ?, ?, ?)",
                                  (user_id,
                                   str(row["date"]),
                                   row["quantity"],
                                   row["food_waste"],
                                   row["temperature"],
                                   row["event"]))
                    conn.commit()

                    st.success("Registration Successful! Please Login.")

                except:
                    st.error("User already exists!")

    if menu == "Login":
        if st.button("Login"):
            c.execute("SELECT id FROM users WHERE unique_id=? AND password=?",
                      (unique_id, password))
            user = c.fetchone()

            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid Credentials")

# ======================================
# DASHBOARD
# ======================================
else:

    st.title("🍽 Smart Kitchen Waste Monitor Dashboard")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # Load user dataset
    query = "SELECT date, quantity, food_waste, temperature, event FROM waste_data WHERE user_id=?"
    data = pd.read_sql_query(query, conn, params=(st.session_state.user_id,))
    data["date"] = pd.to_datetime(data["date"])

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
    st.metric("Model R2 Score", round(accuracy, 2))

    # ======================================
    # PREDICTION
    # ======================================
    st.header("🔮 Waste Prediction")

    quantity = st.slider("Quantity",
                         int(data["quantity"].min()),
                         int(data["quantity"].max()),
                         int(data["quantity"].mean()))

    temperature = st.slider("Temperature",
                            int(data["temperature"].min()),
                            int(data["temperature"].max()),
                            int(data["temperature"].mean()))

    event_option = st.selectbox("Event?", ["No", "Yes"])
    event = 1 if event_option == "Yes" else 0

    input_data = pd.DataFrame([[quantity, temperature, event]],
                              columns=features)

    predicted_waste = model.predict(input_data)[0]

    st.metric("Predicted Waste", round(predicted_waste, 2))

    # Save prediction to database
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
              (st.session_state.user_id,
               quantity,
               temperature,
               event,
               predicted_waste))
    conn.commit()

    # ======================================
    # REPORTS
    # ======================================
    st.header("📊 Reports")

    daily = data.groupby(data["date"].dt.date)["food_waste"].sum()
    st.line_chart(daily)

    # ======================================
    # VISUALIZATION
    # ======================================
    st.header("📉 Quantity vs Waste")

    fig, ax = plt.subplots()
    ax.scatter(data["quantity"], data["food_waste"])
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Waste")
    st.pyplot(fig)

    # ======================================
    # PREDICTION HISTORY
    # ======================================
    st.header("📜 Prediction History")

    history = pd.read_sql_query(
        "SELECT quantity, temperature, event, predicted_waste FROM predictions WHERE user_id=?",
        conn,
        params=(st.session_state.user_id,)
    )

    st.dataframe(history)