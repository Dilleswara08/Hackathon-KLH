# ======================================
# CUSTOM CSS STYLING
# ======================================
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
    }

    /* Main title */
    h1 {
        color: #00FFCC;
        font-family: 'Trebuchet MS', sans-serif;
        text-align: center;
    }

    /* Subheaders */
    h2, h3 {
        color: #FF4B4B;
        font-family: 'Verdana', sans-serif;
    }

    /* Normal text */
    .stMarkdown, .stText {
        color: #FFFFFF;
        font-size: 16px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1C1F26;
    }

    /* Buttons */
    .stButton>button {
        background-color: #00FFCC;
        color: black;
        font-weight: bold;
        border-radius: 10px;
    }

    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)