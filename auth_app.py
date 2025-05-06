"""
RideWise Authentication Module

This module handles user authentication for the RideWise application,
including registration, login, and session management.
"""

import hashlib
import re
import sqlite3
from datetime import datetime

import streamlit as st

# Configure page
st.set_page_config(page_title="RideWise", layout="centered")

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

    /* Override tab underline to blue */
    div[data-baseweb="tab"] button[aria-selected="true"] {
        border-bottom: 3px solid #2575fc;
        color: #2575fc;
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

    /* Remove unwanted top empty box */
    section[data-testid="stTabs"] > div:first-child {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Database connection
DB_CONNECTION = None

def init_db():
    """
    Initialize the SQLite database with user and session tables.
    
    Returns:
        sqlite3.Connection: Database connection object
    """
    connection = sqlite3.connect('user_auth.db', check_same_thread=False)
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    connection.commit()
    return connection

# Initialize global database connection
DB_CONNECTION = init_db()

def hash_password(password):
    """
    Hash a password using SHA-256.
    
    Args:
        password (str): Plain text password
        
    Returns:
        str: Hashed password
    """
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """
    Validate email format using regex.
    
    Args:
        email (str): Email address to validate
        
    Returns:
        bool: True if email is valid, False otherwise
    """
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email) is not None

def validate_password(password):
    """
    Validate password strength.
    
    Args:
        password (str): Password to validate
        
    Returns:
        tuple: (bool, str) indicating if password is valid and a message
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter"
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?/"
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character"
    return True, "Password is valid"

def register_user(username, email, password):
    """
    Register a new user in the database.
    
    Args:
        username (str): User's username
        email (str): User's email
        password (str): User's password
        
    Returns:
        tuple: (bool, str) indicating success/failure and a message
    """
    cursor = DB_CONNECTION.cursor()
    try:
        hashed = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
            (username, email, hashed)
        )
        DB_CONNECTION.commit()
        return True, "Registration successful"
    except sqlite3.IntegrityError as error:
        if "users.username" in str(error):
            return False, "Username already exists"
        if "users.email" in str(error):
            return False, "Email already exists"
        return False, "Registration failed"

def login_user(username_or_email, password):
    """
    Authenticate a user and create a session.
    
    Args:
        username_or_email (str): Username or email for login
        password (str): User's password
        
    Returns:
        tuple: (bool, str) indicating success/failure and username or error message
    """
    cursor = DB_CONNECTION.cursor()
    hashed = hash_password(password)
    if '@' in username_or_email:
        cursor.execute(
            "SELECT id, username FROM users WHERE email = ? AND password = ?", 
            (username_or_email, hashed)
        )
    else:
        cursor.execute(
            "SELECT id, username FROM users WHERE username = ? AND password = ?", 
            (username_or_email, hashed)
        )
    user = cursor.fetchone()
    if user:
        session_id = hashlib.sha256(f"{user[0]}{datetime.now()}".encode()).hexdigest()
        cursor.execute(
            "INSERT INTO sessions (user_id, session_id) VALUES (?, ?)", 
            (user[0], session_id)
        )
        DB_CONNECTION.commit()
        return True, user[1]
    return False, "Invalid username/email or password"
# Session state initialization
for key in ["logged_in", "username", "active_tab", "current_page"]:
    if key not in st.session_state:
        st.session_state[key] = "auth" if key == "current_page" else (False if key == "logged_in" else None)

def logout():
    """Clear session state and log out the user."""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.active_tab = "login"
    st.session_state.current_page = "auth"

def go_to_ml_page():
    """Navigate to machine learning page."""
    st.session_state.current_page = "ml"

def go_to_dl_page():
    """Navigate to deep learning page."""
    st.session_state.current_page = "dl"

def go_to_qml_page():
    """Navigate to quantum machine learning page."""
    st.session_state.current_page = "qml"

def go_to_home():
    """Navigate to home page."""
    st.session_state.current_page = "home"

def main():
    """Main application entry point."""
    if not st.session_state.logged_in:
        display_auth_page()
    else:
        page = st.session_state.current_page
        if page in ["auth", "home"]:
            display_selection_page()
        elif page == "ml":
            display_ml_page()
        elif page == "dl":
            display_dl_page()
        elif page == "qml":
            display_qml_page()

def display_login_tab():
    """Display login tab content."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Login to Your Account")
    login_id = st.text_input("Username or Email", key="login_id")
    login_password = st.text_input("Password", type="password", key="login_password")
    col1, col2 = st.columns([1, 3])
    login_button = col1.button("Login", key="login_btn", type="primary", use_container_width=True)
    col2.markdown(
        "<div style='padding-top: 10px;'><a href='#'>Forgot password?</a></div>", 
        unsafe_allow_html=True
    )
    if login_button:
        if not login_id or not login_password:
            st.error("Please fill in all fields")
        else:
            success, msg = login_user(login_id, login_password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = msg
                st.session_state.current_page = "home"
                st.rerun()
            else:
                st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)

def display_signup_tab():
    """Display signup tab content."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Create a New Account")
    new_username = st.text_input("Username", key="signup_username")
    new_email = st.text_input("Email", key="signup_email")
    new_password = st.text_input("Password", type="password", key="signup_password")
    confirm_password = st.text_input(
        "Confirm Password", 
        type="password",
        key="signup_confirm"
    )
    terms = st.checkbox("I agree to the Terms and Conditions", key="terms")
    signup_button = st.button(
        "Create Account", 
        key="signup_btn",
        type="primary",
        use_container_width=True
    )

    if signup_button:
        if not all([new_username, new_email, new_password, confirm_password]):
            st.error("Please fill in all fields")
        elif not validate_email(new_email):
            st.error("Please enter a valid email address")
        elif new_password != confirm_password:
            st.error("Passwords do not match")
        elif not terms:
            st.error("You must agree to the Terms and Conditions")
        else:
            valid, msg = validate_password(new_password)
            if not valid:
                st.error(msg)
            else:
                success, msg = register_user(new_username, new_email, new_password)
                if success:
                    st.success(msg)
                    st.info("You can now log in with your credentials")
                else:
                    st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)

def display_auth_page():
    """Display authentication page with login and signup tabs."""
    st.markdown(
        '<div class="gradient-text">Welcome to RideWise: Predicting Bike Trip Membership Types</div>', 
        unsafe_allow_html=True
    )
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        display_login_tab()
    with tab2:
        display_signup_tab()

def display_selection_page():
    """Display the main selection page after login."""
    st.header(f"Welcome, {st.session_state.username}!")
    st.subheader("Choose Your Learning Path")
    st.write("Click on any of the buttons below to explore different areas of AI.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Machine Learning", key="ml_btn", on_click=go_to_ml_page)
        st.markdown("Foundational algorithms and statistical models")
    with col2:
        st.button("Deep Learning", key="dl_btn", on_click=go_to_dl_page)
        st.markdown("Neural networks and advanced architectures")
    with col3:
        st.button("Quantum ML", key="qml_btn", on_click=go_to_qml_page)
        st.markdown("Quantum computing applied to ML tasks")
    st.sidebar.button("Logout", key="logout_btn", on_click=logout)

def display_ml_page():
    """Display machine learning page with redirect to external app."""
    other_app_url = "http://localhost:8502/"
    st.markdown(
        f'<meta http-equiv="refresh" content="0;url={other_app_url}">', unsafe_allow_html=True)
    st.button("\u2190 Back to Home", key="back_to_home", on_click=go_to_home)
    st.sidebar.button("Logout", key="logout_btn_ml", on_click=logout)

def display_dl_page():
    """Display deep learning information page."""
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
    st.image("deep-learning.png")
    st.button("\u2190 Back to Home", key="back_to_home_dl", on_click=go_to_home)
    st.sidebar.button("Logout", key="logout_btn_dl", on_click=logout)

def display_qml_page():
    """Display quantum machine learning information page."""
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
    st.image("qml.png")
    st.button("\u2190 Back to Home", key="back_to_home_qml", on_click=go_to_home)
    st.sidebar.button("Logout", key="logout_btn_qml", on_click=logout)

if __name__ == "__main__":
    main()
