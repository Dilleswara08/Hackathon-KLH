# SMART KITCHEN WASTE MONITOR

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import datetime

# -----------------------------
# 1️⃣ Generate Synthetic Dataset
# -----------------------------

np.random.seed(42)

days = 365

data = pd.DataFrame({
    "date": pd.date_range(start="2024-01-01", periods=days),
    "sales": np.random.randint(200, 500, days),
    "temperature": np.random.randint(20, 40, days),
    "rain": np.random.randint(0, 2, days),  # 0 = No rain, 1 = Rain
    "event": np.random.randint(0, 2, days)  # 0 = No event, 1 = Event
})

# Food prepared = sales + buffer (5% to 15%)
data["food_prepared"] = data["sales"] * np.random.uniform(1.05, 1.15, days)

# Waste depends on weather + event + randomness
data["food_waste"] = (
    0.1 * data["sales"]
    + 5 * data["rain"]
    - 2 * data["event"]
    + np.random.normal(0, 5, days)
)

data["food_waste"] = data["food_waste"].abs()

# -----------------------------
# 2️⃣ Data Preparation
# -----------------------------

features = ["sales", "temperature", "rain", "event"]
X = data[features]
y = data["food_waste"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3️⃣ Train Regression Model
# -----------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 4️⃣ Model Evaluation
# -----------------------------

predictions = model.predict(X_test)

print("Model Performance:")
print("R2 Score:", r2_score(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))

# -----------------------------
# 5️⃣ Waste Prediction Function
# -----------------------------

def predict_waste(sales, temperature, rain, event):
    input_data = np.array([[sales, temperature, rain, event]])
    predicted = model.predict(input_data)[0]
    return max(predicted, 0)

# -----------------------------
# 6️⃣ Order Recommendation Engine
# -----------------------------

def recommend_order(expected_sales, temperature, rain, event):
    predicted_waste = predict_waste(expected_sales, temperature, rain, event)
    
    recommended_order = expected_sales - predicted_waste
    
    # Extra business logic
    if event == 1:
        recommended_order *= 1.10  # increase 10% for events
    if rain == 1:
        recommended_order *= 0.95  # reduce 5% if raining
    
    return round(recommended_order, 2), round(predicted_waste, 2)

# -----------------------------
# 7️⃣ Demo Prediction
# -----------------------------

print("\n--- Demo Prediction ---")

expected_sales = 400
temperature = 32
rain = 1
event = 0

order, waste = recommend_order(expected_sales, temperature, rain, event)

print("Expected Sales:", expected_sales)
print("Predicted Waste:", waste)
print("Recommended Order Quantity:", order)