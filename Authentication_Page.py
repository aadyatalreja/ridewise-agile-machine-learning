"""
RideWise Authentication Module

This module handles user authentication for the RideWise application,
including registration, login, and session management.
"""

import hashlib
import re
import sqlite3
from datetime import datetime
import os
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.state import SessionStateProxy

# Configure page
st.set_page_config(page_title="RideWise Authentication", layout="centered")

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
        # Store session data
        st.session_state.user_id = user[0]
        st.session_state.session_id = session_id
        return True, user[1]
    return False, "Invalid username/email or password"

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
                # Save to session state for home_page.py to access
                st.session_state.logged_in = True
                st.session_state.username = msg
                # Redirect to home page
                st.switch_page("pages/home_page.py")
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

def main():
    """Main application entry point."""
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    
    if not st.session_state.logged_in:
        display_auth_page()
    else:
        # Redirect to the home page if logged in
        try:
            st.switch_page("home_page")
        except Exception as e:
            st.error(f"Error redirecting to home page: {e}")
            st.info("Make sure home_page.py exists in the same directory and you're using Streamlit version 1.10.0 or higher")
            
            # Fallback method
            st.write("You are logged in! Please click the button below to go to the home page manually.")
            if st.button("Go to Home Page"):
                st.switch_page("home_page")

if __name__ == "__main__":
    main()