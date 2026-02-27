import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
from datetime import datetime
import streamlit as st
import requests
import streamlit.components.v1 as components


st.set_page_config(page_title="Food Waste Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1046/1046784.png", width=80)
st.sidebar.title("📊 Navigation")
app_mode = st.sidebar.radio("Go to", ["🏠 Home", "📄 About", "📊 Predictions", "🥕 Ingredients Overview", "📬 Contact"])

# Load model and features if needed
try:
    model = joblib.load("ingredient_waste_model.pkl")
    features = joblib.load("model_features.pkl")
except:
    model, features = None, []

# --- HOME ---
if app_mode == "🏠 Home":
    from streamlit_extras.let_it_rain import rain

    # 🎯 Centered Title
    st.markdown("<h1 style='text-align: center; color: #F63366;'>🍽️ Food Waste Prediction System</h1>", unsafe_allow_html=True)

    # 📸 Background / Banner Image
    st.image("background_img.jpg", use_column_width=True)

    # 🌧️ Raining Tomatoes Animation
    'rain(emoji="🍅", font_size=42, falling_speed=5, animation_length="infinite")'

    # 💬 Welcome Message
    st.markdown("""
    <div style='text-align: center; font-size:18px; padding:10px 0 20px 0;'>
    Welcome to the <b>Food Waste Forecasting System</b> — your smart kitchen assistant to reduce ingredient waste and save money! 🌱
    </div>
    """, unsafe_allow_html=True)

    # 📊 KPI Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Waste Saved", "120 kg", "+30%")
    col2.metric("Cost Reduction", "₹ 18,500", "+15%")
    col3.metric("Model Accuracy", "92.4%", "↑")

    # ✅ Key Features
    st.markdown("""
    ---
    ### ✅ Key Features
    - 📈 Predict unsold ingredients
    - 💰 Optimize inventory cost
    - 📅 Use calendar features (holidays, weather)
    - 🧠 Machine learning powered forecasts
    """)

    # 🚀 How it Works
    st.markdown("""
    ---
    ### 🚀 How it Works
    1. Upload historical data (sales + purchase)
    2. Merge with recipe & calendar info
    3. Train a model to predict unsold quantities
    4. Use predictions to reduce food waste
    """)

    # 🔎 Navigation
    st.markdown("---")
    st.info("🔎 Use the sidebar to explore predictions, ingredients overview, and project details.")

# --- ABOUT ---
elif app_mode == "📄 About":
    st.markdown("<h2 style='text-align: center; color:#F63366;'>🔍 About the Project</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: justify; font-size: 16px;'>
    <br>
    <b>Food Waste Prediction in Restaurants</b> is a smart forecasting system designed to reduce ingredient waste and optimize cost in restaurant kitchens.  
    Using machine learning models trained on historical sales, inventory, and calendar data, it predicts which ingredients are likely to remain unsold on a given day.  
    <br><br>
    The goal is to help restaurants reduce over-purchasing, minimize food wastage, and increase operational efficiency. 🌿
    </div>
    """, unsafe_allow_html=True)

    # Dataset Info Section
    with st.expander("📁 Dataset Overview"):
        st.markdown("""
        - **Sales Data**: Daily dish-wise sales log  
        - **Waste Data**: Ingredient purchase, usage & wastage tracking  
        - **Calendar Data**: Holiday, weather & weekend impact  
        - **Recipe Mapping**: Dish-to-ingredient quantity mapping  
        """)

    # Tech Used
    with st.expander("🧠 Technologies Used"):
        st.markdown("""
        - **Python**: Data processing and modeling  
        - **Pandas & NumPy**: Data wrangling  
        - **Scikit-learn**: Machine learning model  
        - **Streamlit**: Dashboard frontend  
        - **Matplotlib, Seaborn, Plotly**: Visualization  
        """)

    # 🎯 Team Section

    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color:#F63366;'>👨‍💻 Project Contributors</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        
        st.markdown("**Harsh Awasthi**")
        

        
        st.markdown("**Divyanshu Ranjan**")
        

    with col2:
        
        st.markdown("**Awanish Bhatt**")
        

        
        st.markdown("**Akash Nigam**")
        

    # Optional Fun Facts or Credits
    st.markdown("---")
    st.success("💡 Fun Fact: Over 40% of food produced globally is wasted. This project is our small step toward solving that.")






# Prediction Page

elif app_mode == "📊 Predictions":
    st.markdown("<h2 style='text-align: center; color:#F63366;'>📊 Ingredient Waste Prediction</h2>", unsafe_allow_html=True)
    st.markdown("Upload a CSV file to forecast unsold ingredient quantities using your trained machine learning model. 🍽️")

    # File Upload
    uploaded_file = st.file_uploader("📤 Upload a CSV File", type=["csv"], help="File must match model features")

    if uploaded_file is not None:
        st.markdown("---")
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df.head())

        # Encode and align with model features
        df_encoded = pd.get_dummies(df)
        for col in features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[features]

        if model:
            with st.spinner("🔍 Running Predictions..."):
                predictions = model.predict(df_encoded)
                df["Predicted_Unsold_kg"] = predictions.round(2)

            st.success("✅ Prediction Completed Successfully!")

            # Summary dashboard
            st.markdown("### 📊 Summary Statistics")
            total_predicted = df["Predicted_Unsold_kg"].sum()
            avg_predicted = df["Predicted_Unsold_kg"].mean()
            top_ingredient = df.loc[df["Predicted_Unsold_kg"].idxmax(), "Ingredient"]

            col1, col2, col3 = st.columns(3)
            col1.metric("📦 Total Predicted Waste", f"{total_predicted:.2f} kg")
            col2.metric("📉 Average Waste / Ingredient", f"{avg_predicted:.2f} kg")
            col3.metric("🔥 Highest Waste Ingredient", top_ingredient)

            # Main results table
            st.markdown("---")
            st.subheader("📦 Prediction Results Table")
            styled_df = df[["Ingredient", "Predicted_Unsold_kg"]].sort_values(by="Predicted_Unsold_kg", ascending=False)
            st.dataframe(
                styled_df.style.background_gradient(cmap="YlOrRd", subset=["Predicted_Unsold_kg"])
            )

            # Chart: Predicted vs Actual (if available)
            if "Unsold_Quantity" in df.columns:
                st.markdown("---")
                st.subheader("📈 Predicted vs Actual Unsold Quantity")
                fig = px.scatter(
                    df,
                    x="Unsold_Quantity",
                    y="Predicted_Unsold_kg",
                    color="Ingredient",
                    title="Predicted vs Actual Unsold Quantity",
                    labels={
                        "Unsold_Quantity": "Actual Unsold (kg)",
                        "Predicted_Unsold_kg": "Predicted Unsold (kg)"
                    }
                )
                st.plotly_chart(fig)

            # Chart: Top N wasted ingredients
            st.markdown("### 🧾 Top 10 Ingredients by Predicted Waste")
            top10 = styled_df.head(10)
            fig2 = px.bar(
                top10,
                x="Ingredient",
                y="Predicted_Unsold_kg",
                color="Predicted_Unsold_kg",
                title="Top 10 Wasted Ingredients (Predicted)",
                labels={"Predicted_Unsold_kg": "Waste (kg)"}
            )
            st.plotly_chart(fig2)

            # Download option
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Prediction Results", csv, "predicted_unsold.csv", "text/csv")

            # Optional: Feature explanation
            with st.expander("ℹ️ What do these features mean?"):
                st.markdown("""
                - **Ingredient**: The raw material or food item used in the dish.
                - **Predicted_Unsold_kg**: Estimated leftover quantity after sales.
                - **Unsold_Quantity**: Actual recorded waste (if available).
                - **Other columns**: Encoded features like weather, day, dish types, etc.
                """)
        else:
            st.error("❌ Model is not loaded. Please ensure `model_features.pkl` is available.")

# --- INGREDIENTS OVERVIEW ---
elif app_mode == "🥕 Ingredients Overview": 
    st.markdown("<h2 style='text-align: center; color:#228B22;'>🥕 Ingredients Overview</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Explore key ingredients used across dishes and monitor their overall usage. Understanding ingredient patterns helps in forecasting demand and minimizing waste. 📉🍲
    """)

    # Ingredient usage data
    ingredient_list = ["Chicken", "Tomatoes", "Onions", "Paneer", "Butter", "Rice", "Dal"]
    usage_kg = [120, 95, 110, 60, 85, 130, 105]

    # Summary cards
    st.markdown("### 📊 Ingredient Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌾 Top Used", f"{ingredient_list[5]} ({usage_kg[5]} kg)")
    col2.metric("🍅 Most Common Veg", f"{ingredient_list[1]} ({usage_kg[1]} kg)")
    col3.metric("🍗 Most Common Non-Veg", f"{ingredient_list[0]} ({usage_kg[0]} kg)")

    # Plotly bar chart
    st.markdown("### 📈 Usage Distribution")
    fig = px.bar(
        x=ingredient_list,
        y=usage_kg,
        color=usage_kg,
        labels={'x': 'Ingredient', 'y': 'Usage (kg)'},
        title="Ingredient Usage Snapshot",
        color_continuous_scale="YlOrRd"
    )
    fig.update_layout(xaxis_title="Ingredient", yaxis_title="Total Usage (kg)")
    st.plotly_chart(fig)

    # Optional food image for visual appeal
    st.image("https://cdn.pixabay.com/photo/2016/03/05/19/02/hamburger-1238246_960_720.jpg", caption="Efficient ingredient use = less waste 🍔", use_column_width=True)

# --- CONTACT ---