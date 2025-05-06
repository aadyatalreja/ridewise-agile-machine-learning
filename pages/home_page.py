"""
RideWise Home Page Module

This module handles the home page and learning path selection
for the RideWise application after user authentication.
"""

import streamlit as st
import os
import sqlite3
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Configure page
st.set_page_config(page_title="RideWise Home", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
    /* Animated gradient for the title */
    .gradient-text {
        font-size: 2.5em;
        font-weight: 800;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-flow 3s infinite alternate;
    }

    @keyframes gradient-flow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    /* Glass effect container */
    .glass-box {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 3rem 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-top: 2rem;
    }

    /* Button styling */
    div.stButton > button {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75em 1em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Database connection
DB_CONNECTION = None

def init_db():
    """Initialize the SQLite database connection."""
    connection = sqlite3.connect('user_auth.db', check_same_thread=False)
    return connection

DB_CONNECTION = init_db()

def verify_session():
    """Verify if the user has a valid session."""
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        return False
    if "user_id" in st.session_state and "session_id" in st.session_state:
        cursor = DB_CONNECTION.cursor()
        cursor.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND session_id = ?",
            (st.session_state.user_id, st.session_state.session_id)
        )
        if cursor.fetchone():
            return True
    return False

def logout():
    """Clear session state and redirect to auth page."""
    if "user_id" in st.session_state and "session_id" in st.session_state:
        cursor = DB_CONNECTION.cursor()
        cursor.execute(
            "DELETE FROM sessions WHERE user_id = ? AND session_id = ?", 
            (st.session_state.user_id, st.session_state.session_id)
        )
        DB_CONNECTION.commit()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    try:
        st.switch_page("Authentication_Page.py")
    except:
        st.rerun()

def go_to_ml_page():
    st.session_state.current_page = "ml"
    # st.rerun()

def go_to_dl_page():
    st.session_state.current_page = "dl"
    # st.rerun()

def go_to_qml_page():
    st.session_state.current_page = "qml"
    # st.rerun()

def go_to_qnn_page():
    st.session_state.current_page = "qnn"
    # st.rerun()

def go_to_home():
    st.session_state.current_page = "home"
    # st.rerun()

def display_selection_page():
    """Display the main selection page after login."""
    st.markdown(
        '<div class="gradient-text">RideWise: Predicting Bike Trip Membership Types</div>', 
        unsafe_allow_html=True
    )
    st.header(f"Welcome, {st.session_state.username}!")
    st.subheader("Choose Your Learning Path")
    st.write("Click on any of the buttons below to explore different areas of AI.")

    # First row: ML and DL
    col1, col2 = st.columns(2)
    with col1:
        st.button("Machine Learning", key="ml_btn", on_click=go_to_ml_page)
        st.markdown("Foundational algorithms and statistical models")
    with col2:
        st.button("Deep Learning", key="dl_btn", on_click=go_to_dl_page)
        st.markdown("Neural networks and advanced architectures")

    # Second row: QML and QNN
    col3, col4 = st.columns(2)
    with col3:
        st.button("Quantum Machine Learning", key="qml_btn", on_click=go_to_qml_page)
        st.markdown("Quantum computing applied to ML tasks")
    with col4:
        st.button("Quantum Neural Networks", key="qnn_btn", on_click=go_to_qnn_page)
        st.markdown("Quantum Neural Networks (QNNs) deep dive")

    st.sidebar.button("Logout", key="logout_btn", on_click=logout)

def display_ml_page():
    st.title("Machine Learning")
    st.write("Explore machine learning algorithms and statistical models.")

    st.subheader("Popular Algorithms")
    st.write("""
    - Linear Regression
    - Logistic Regression
    - Decision Trees
    - Random Forests
    - Support Vector Machines
    - K-Means Clustering
    - Gradient Boosting
    """)
    if os.path.exists("ml.jpeg"):
        st.image("ml.jpeg")
    st.button("\u2190 Back to Home", key="back_to_home_ml", on_click=go_to_home)
    st.sidebar.button("Logout", key="logout_btn_ml", on_click=logout)

def display_dl_page():
    st.title("Deep Learning")
    st.write("Explore neural networks and deep learning architectures.")
    st.subheader("Popular Architectures")
    st.write("""
    - CNNs
    - RNNs
    - LSTMs
    - GANs
    - Transformers
    - Autoencoders
    """)
    if os.path.exists("deep-learning.png"):
        st.image("deep-learning.png")
    st.button("\u2190 Back to Home", key="back_to_home_dl", on_click=go_to_home)
    st.sidebar.button("Logout", key="logout_btn_dl", on_click=logout)

def display_qml_page():
    st.title("Quantum Machine Learning")
    st.write("Quantum computing meets machine learning.")
    st.subheader("Key Concepts")
    st.write("""
    - Quantum Neural Networks
    - Quantum SVMs
    - Variational Quantum Eigensolvers
    - Quantum Annealing
    - Quantum GANs
    - Quantum RL
    """)
    if os.path.exists("qml.png"):
        st.image("qml.png")
    st.button("\u2190 Back to Home", key="back_to_home_qml", on_click=go_to_home)
    st.sidebar.button("Logout", key="logout_btn_qml", on_click=logout)

def display_qnn_page():
    st.title("Quantum Neural Networks (QNN)")
    st.write("Explore the intersection of neural networks and quantum computing.")
    st.subheader("Key Topics")
    st.write("""
    - Parameterized Quantum Circuits
    - Hybrid Classical-Quantum Architectures
    - QNN Optimization
    - Applications in Chemistry, Finance, and Cryptography
    """)
    if os.path.exists("qnn.png"):
        st.image("qnn.png")
    st.button("\u2190 Back to Home", key="back_to_home_qnn", on_click=go_to_home)
    st.sidebar.button("Logout", key="logout_btn_qnn", on_click=logout)

def main():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    if not verify_session():
        st.warning("You are not logged in. Please log in first.")
        try:
            st.switch_page("Authentication_Page.py")
        except:
            st.error("Could not automatically redirect. Please go back to the authentication page manually.")
            if st.button("Go to Login Page"):
                st.switch_page("Authentication_Page.py")
        return
    page = st.session_state.current_page
    if page == "home":
        display_selection_page()
    elif page == "ml":
        if st.button("Go to ML App"):
            st.switch_page("pages/Machine_Learning.py")
        display_ml_page()
    elif page == "dl":
        if st.button("Go to DL App"):
            st.switch_page("pages/Deep_Learning.py")
        display_dl_page()
    elif page == "qml":
        if st.button("Go to QML App"):
            st.switch_page("pages/Quantum_Machine_Learning.py")
        display_qml_page()
    elif page == "qnn":
        if st.button("Go to QNN App"):
            st.switch_page("pages/Quantum_Neural_Networks.py")
        display_qnn_page()

if __name__ == "__main__":
    main()
