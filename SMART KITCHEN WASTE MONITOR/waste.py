import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Table
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Smart Kitchen Waste Monitor", layout="wide")

st.title("🍽 Smart Kitchen Waste Monitor")
st.caption("AI-Powered Waste Prediction & Order Optimization System")

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("⚙️ Data Options")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset (Optional)", type=["csv"])

# -----------------------------------
# DATA INGESTION
# -----------------------------------
def generate_synthetic_data():
    np.random.seed(42)
    days = 365
    df = pd.DataFrame({
        "date": pd.date_range(start="2026-01-01", periods=days),
        "quantity": np.random.randint(200, 500, days),
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
    st.success("Custom dataset loaded successfully.")
else:
    data = generate_synthetic_data()
    st.info("Using synthetic demo dataset.")

st.write("### 📊 Dataset Preview")
st.dataframe(data.head())

# -----------------------------------
# MODEL TRAINING
# -----------------------------------
features = ["sales", "temperature", "rain", "event"]
target = "food_waste"

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("R2 Score", round(r2, 2))
col2.metric("MAE", round(mae, 2))

# -----------------------------------
# PREDICTION SECTION
# -----------------------------------
st.header("🔮 Predict Waste & Optimize Order")

col1, col2 = st.columns(2)

with col1:
    sales = st.slider("Expected Sales", 200, 600, 400)
    temperature = st.slider("Temperature (°C)", 20, 45, 30)

with col2:
    rain = st.selectbox("Rain?", [0, 1])
    event = st.selectbox("Local Event?", [0, 1])

input_df = pd.DataFrame([[sales, temperature, rain, event]],
                        columns=features)

predicted_waste = model.predict(input_df)[0]

recommended_order = sales - predicted_waste
if event == 1:
    recommended_order *= 1.10
if rain == 1:
    recommended_order *= 0.95

st.subheader("📊 Prediction Results")
st.write("Predicted Waste:", round(predicted_waste, 2))
st.write("Recommended Order Quantity:", round(recommended_order, 2))

# -----------------------------------
# VISUALIZATION
# -----------------------------------
st.header("📉 Waste Analysis")

fig, ax = plt.subplots()
ax.scatter(data["sales"], data["food_waste"])
ax.set_xlabel("Sales")
ax.set_ylabel("Food Waste")
ax.set_title("Sales vs Waste")
st.pyplot(fig)

# -----------------------------------
# IMPACT REPORT
# -----------------------------------
st.header("📑 Waste Reduction Impact Report")

baseline_waste = 0.15 * sales
improvement = baseline_waste - predicted_waste

st.write("Baseline Waste (15% Rule):", round(baseline_waste, 2))
st.write("AI Predicted Waste:", round(predicted_waste, 2))
st.write("Estimated Reduction:", round(improvement, 2))

st.success("🎉 AI system helps reduce food waste and increase profitability.")

# -----------------------------------
# PDF GENERATION
# -----------------------------------
def generate_pdf():
    file_path = "Waste_Report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Smart Kitchen Waste Monitor Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Predicted Waste: {round(predicted_waste,2)}", styles["Normal"]))
    elements.append(Paragraph(f"Recommended Order: {round(recommended_order,2)}", styles["Normal"]))
    elements.append(Paragraph(f"Estimated Reduction: {round(improvement,2)}", styles["Normal"]))

    doc.build(elements)
    return file_path

if st.button("📥 Download Impact Report (PDF)"):
    pdf_path = generate_pdf()
    with open(pdf_path, "rb") as f:
        st.download_button("Download Report", f, file_name="Waste_Report.pdf")