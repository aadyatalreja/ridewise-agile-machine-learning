import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import plotly.graph_objects as go
# Import additional classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Import Google's Generative AI library for Gemini
import google.generativeai as genai
# Import OpenAI for fallback
# Custom CSS Styling - removed glass effect
st.set_page_config(page_title="RideWise ML Analysis", layout="centered")

# Apply styling without glass-box
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

    /* Standard section styling instead of glass */
    .section-container {
        padding: 1.5rem 0;
        margin: 1rem 0;
        border-bottom: 1px solid #f0f0f0;
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
    
    /* User message style */
    .user-message {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    
    /* Bot message style */
    .bot-message {
        background: #e9e9e9;
        color: #333;
        border-radius: 18px 18px 18px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-right: auto;
        word-wrap: break-word;
    }
    
    
    /* Chat input field */
    .chat-input input {
        flex-grow: 1;
        border-radius: 20px;
        border: 1px solid #ddd;
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
    }
    
    /* Chat messages wrapper */
    .chat-messages {
        padding: 1rem;
        overflow-y: auto;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
    
    /* User message style with improved gradient */
    .user-message {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 0.8rem 1rem;
        margin: 0.25rem 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.3s ease-out;
    }
    
    
    /* Message fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Chat header */
    .chat-header {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px 12px 0 0;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .chat-header-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def loadData():
    df = pd.read_csv("2010-capitalbikeshare-tripdata.csv")
    return df

# Basic preprocessing required for all the models.
def preprocessing(df):
    X = df.iloc[:, [0, 3, 5]].values
    y = df.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y.flatten())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    return X_train, X_test, y_train, y_test, le


@st.cache_resource
def decisionTree(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, tree


@st.cache_resource
def Knn_Classifier(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, clf


@st.cache_resource
def svm_classifier(X_train, X_test, y_train, y_test):
    # Use SGDClassifier instead of SVC - much faster
    from sklearn.linear_model import SGDClassifier    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train using SGD with hinge loss (equivalent to linear SVM but faster)
    clf = SGDClassifier(loss='hinge', penalty='l2', max_iter=100, tol=1e-3, random_state=0)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, clf, scaler

@st.cache_resource
def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, clf

@st.cache_resource
def random_forest_classifier(X_train, X_test, y_train, y_test):
    # Optimize Random Forest parameters for speed
    clf = RandomForestClassifier(
        n_estimators=50,        # Reduced from 100
        max_depth=10,           # Limit tree depth
        min_samples_split=5,    # Require more samples to split
        min_samples_leaf=2,     # Require more samples in leaves
        max_features='sqrt',    # Use sqrt of features (faster)
        n_jobs=-1,              # Use all available cores
        random_state=0)
    
    # Sample the data if it's too large
    max_samples = min(2000, len(X_train))
    if len(X_train) > max_samples:
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        X_train_sampled = X_train[indices]
        y_train_sampled = y_train[indices]
    else:
        X_train_sampled = X_train
        y_train_sampled = y_train
    # Train the model
    clf.fit(X_train_sampled, y_train_sampled)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, clf

@st.cache_resource
def logistic_regression_classifier(X_train, X_test, y_train, y_test):
    # Scale data for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, clf, scaler

def accept_user_data():
    duration = st.text_input("Enter the Duration: ")
    start_station = st.text_input("Enter the start station number: ")
    end_station = st.text_input("Enter the end station number: ")
    user_prediction_data = np.array([duration, start_station, end_station]).reshape(1, -1)
    return user_prediction_data

@st.cache_resource
def showMap():
    plotData = pd.read_csv("Trip history with locations.csv")
    Data = pd.DataFrame()
    Data['lat'] = plotData['lat']
    Data['lon'] = plotData['lon']
    return Data

def compare_all_models(X_train, X_test, y_train, y_test):
    st.markdown('<div class="gradient-text">Machine Learning Model Comparison Dashboard</div>', unsafe_allow_html=True)
    st.write("Comparing the performance of all machine learning models")
    # Create a progress bar to show the model training progress
    progress_bar = st.progress(0)
    # Create a dictionary to store all model accuracies
    all_models = {
        "Decision Tree": None,
        "K-Nearest Neighbors": None,
        "SVM": None,
        "Naive Bayes": None,
        "Random Forest": None,
        "Logistic Regression": None
    }
    
    # Train all models and get accuracies
    with st.spinner("Training Decision Tree..."):
        score, _, _ = decisionTree(X_train, X_test, y_train, y_test)
        all_models["Decision Tree"] = score
        progress_bar.progress(14)
    
    with st.spinner("Training K-Nearest Neighbors..."):
        score, _, _ = Knn_Classifier(X_train, X_test, y_train, y_test)
        all_models["K-Nearest Neighbors"] = score
        progress_bar.progress(42)
    
    with st.spinner("Training SVM..."):
        score, _, _, _ = svm_classifier(X_train, X_test, y_train, y_test)
        all_models["SVM"] = score
        progress_bar.progress(56)
    
    with st.spinner("Training Naive Bayes..."):
        score, _, _ = naive_bayes_classifier(X_train, X_test, y_train, y_test)
        all_models["Naive Bayes"] = score
        progress_bar.progress(70)
    
    with st.spinner("Training Random Forest..."):
        score, _, _ = random_forest_classifier(X_train, X_test, y_train, y_test)
        all_models["Random Forest"] = score
        progress_bar.progress(84)
    
    with st.spinner("Training Logistic Regression..."):
        score, _, _, _ = logistic_regression_classifier(X_train, X_test, y_train, y_test)
        all_models["Logistic Regression"] = score
        progress_bar.progress(100)
    
    # Convert dictionary to dataframe for display
    df_models = pd.DataFrame(list(all_models.items()), columns=['Model', 'Accuracy (%)'])
    
    # Sort by accuracy
    df_models = df_models.sort_values(by='Accuracy (%)', ascending=False)
    
    # Display the model comparison table
    st.subheader("Model Accuracy Comparison")
    st.dataframe(df_models, use_container_width=True)
    
    # Create a bar chart comparing model accuracies
    fig = px.bar(
        df_models, 
        x='Model', 
        y='Accuracy (%)',
        color='Accuracy (%)',
        color_continuous_scale='viridis',
        title='Model Accuracy Comparison',
        text='Accuracy (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify the best performing model
    best_model = df_models.iloc[0]['Model']
    best_accuracy = df_models.iloc[0]['Accuracy (%)']
    
    st.success(f"The best performing model is **{best_model}** with an accuracy of **{best_accuracy:.2f}%**")
    
    # Add recommendations based on model performance
    st.subheader("Recommendations")
    st.write("""
    - **High Accuracy Models:** Consider using these for your production environment
    - **Low Accuracy Models:** May need parameter tuning or more features
    - **Model Selection:** Choose based on both accuracy and interpretability needs
    """)
    
    if st.button("Return to Main Page"):
        st.session_state.page = "main"
        st.rerun()

# Setup LLM assistant with multiple provider options
def setup_llm_assistant():
    """
    Sets up an LLM assistant with multiple provider options.
    Tries Gemini first, then OpenAI, then falls back to local.
    """
    # First try Gemini
    gemini_model = setup_gemini()
    if gemini_model:
        return {"provider": "gemini", "model": gemini_model}
    
    # If Gemini fails, try OpenAI
    try:
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY
            # Test the OpenAI connection
            test_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            if test_response:
                st.sidebar.success("Connected to OpenAI (fallback mode)")
                return {"provider": "openai", "model": "gpt-3.5-turbo"}
    except Exception as e:
        st.sidebar.warning(f"OpenAI fallback failed: {str(e)}")
    
    # If both fail, use local fallback
    st.sidebar.warning("Using basic local assistant (all LLM services unavailable)")
    return {"provider": "local", "model": None}

# Set up Gemini with fallback options
def setup_gemini():
    """
    Sets up the Gemini API with fallback options for different models.
    Returns a working model or None if all attempts fail.
    """
    try:
        # In production, use st.secrets to manage API keys
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            st.warning("Gemini API key not found in secrets. Assistant features are disabled.")
            return None
            
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Set up the Gemini model with specific parameters
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40
        }
        
        # List of models to try in order of preference
        model_options = [
            "gemini-1.5-pro",    # Try the newer model first
            "gemini-pro",        # Fall back to original model
            "gemini-1.0-pro",    # Another possible fallback
        ]
        
        # Try each model in sequence until one works
        for model_name in model_options:
            try:
                st.sidebar.info(f"Attempting to connect to {model_name}...")
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config
                )
                
                # Test the model with a simple prompt to verify it's working
                test_response = model.generate_content("Hello")
                if test_response:
                    st.sidebar.success(f"Successfully connected to {model_name}")
                    return model
            except Exception as model_error:
                st.sidebar.warning(f"Failed to initialize {model_name}: {str(model_error)}")
                continue
        
        # If we get here, all models failed
        st.error("All Gemini model options failed. Assistant features are disabled.")
        return None
        
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return None

# Function to get response from Gemini with improved error handling
def get_gemini_response(model, user_input):
    """Get response from Gemini with improved error handling"""
    if model is None:
        return "I'm sorry, the assistant is currently unavailable due to API configuration issues. Please try again later or contact support."
        
    # Define the system prompt that guides the chatbot's behavior
    system_prompt = """
    You are RideWise Assistant, a helpful chatbot for the RideWise bike trip prediction application.
    Provide concise, friendly responses about:
    - How to use the RideWise app
    - Information about bike sharing systems
    - Explanations of machine learning models used in the app (Decision Tree, KNN, SVM, Naive Bayes, Random Forest, Logistic Regression)
    - How to interpret model results and predictions
    - How to input data for predictions
    
    Keep responses brief, informative, and in a friendly tone.
    """
    
    try:
        # Set up chat
        chat = model.start_chat(history=[])
        
        # Get response
        response = chat.send_message(f"System: {system_prompt}\n\nUser: {user_input}")
        return response.text
    except Exception as e:
        # Provide more helpful error message based on the exception
        error_message = str(e)
        if "quota" in error_message.lower() or "rate" in error_message.lower():
            return "I'm sorry, the assistant is temporarily unavailable due to usage limits. Please try again in a few minutes."
        elif "connect" in error_message.lower() or "timeout" in error_message.lower():
            return "I'm sorry, there seems to be a connection issue with the assistant service. Please check your internet connection and try again."
        else:
            return f"I'm sorry, I encountered an error. Please try again later or contact support if the issue persists."

# Local fallback assistant for when API services are unavailable
def local_fallback_assistant(user_input):
    """
    Provides basic responses when Gemini API is unavailable.
    This function uses simple keyword matching for common questions.
    """
    user_input = user_input.lower()
    
    # Dictionary of common questions and answers
    faq = {
        "help": "RideWise helps predict bike membership types based on trip data. You can select different machine learning models from the sidebar and compare their performance.",
        
        "model": "RideWise uses several machine learning models: Decision Tree, K-Nearest Neighbors, SVM, Naive Bayes, Random Forest, and Logistic Regression. Each model has different strengths for predicting membership types.",
        
        "accuracy": "Model accuracy varies, but typically ranges from 70-90% depending on the algorithm used. You can see detailed accuracy metrics by selecting a specific model or using the 'Compare All Models' button.",
        
        "predict": "To make a prediction, select a model from the sidebar, check the 'Want to predict on your own input?' box, enter the required data, and click 'Predict'.",
        
        "data": "The application uses bike sharing trip data including duration, start station, and end station to predict membership types. You can view a sample of the raw data by checking 'Show Raw Data'.",
        
        "decision tree": "Decision Trees are simple but powerful models that make decisions based on feature values. They're easy to interpret but can overfit without proper constraints.",
        
        "knn": "K-Nearest Neighbors classifies data points based on the majority class of their k nearest neighbors. It's simple but can be computationally expensive for large datasets.",
        
        "svm": "Support Vector Machines find an optimal hyperplane to separate different classes. They work well with complex data but may require careful parameter tuning.",
        
        "naive bayes": "Naive Bayes is a probabilistic model based on Bayes' theorem. It's fast and works well with high-dimensional data but assumes feature independence.",
        
        "random forest": "Random Forest combines multiple decision trees to improve accuracy and reduce overfitting. It's robust but less interpretable than a single decision tree.",
        
        "logistic regression": "Logistic Regression estimates probabilities of class membership. It's easy to interpret and efficient but may underperform with complex nonlinear relationships."
    }
    
    # Look for keyword matches
    for keyword, response in faq.items():
        if keyword in user_input:
            return response
    
    # Default response if no keywords match
    return "I'm a simple fallback assistant for RideWise. I can answer basic questions about the application and its models. Try asking about specific models, prediction, or how to use the app."

# Unified function to get responses from any available LLM provider
def get_assistant_response(llm_config, user_input):
    """Unified function to get responses from any available LLM provider"""
    provider = llm_config["provider"]
    
    # Define the system prompt for all providers
    system_prompt = """
    You are RideWise Assistant, a helpful chatbot for the RideWise bike trip prediction application.
    Provide concise, friendly responses about:
    - How to use the RideWise app
    - Information about bike sharing systems
    - Explanations of machine learning models used in the app
    - How to interpret model results and predictions
    - How to input data for predictions
    
    Keep responses brief, informative, and in a friendly tone.
    """
    
    if provider == "gemini":
        # Use the existing Gemini function
        return get_gemini_response(llm_config["model"], user_input)
    
    elif provider == "openai":
        try:
            response = openai.chat.completions.create(
                model=llm_config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Assistant Error: {str(e)}"
    
    else:  # local fallback
        return local_fallback_assistant(user_input)

# Chat interface with unified assistant approach
def display_chat_interface():
    st.sidebar.markdown("---")
    st.sidebar.subheader("RideWise Assistant")
    
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Setup the assistant with fallback options
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config = setup_llm_assistant()
    
    # Chat header
    st.sidebar.markdown("""
        <div class="chat-header">
            <div class="chat-header-title">
                <span>🤖</span>
                <span>RideWise Assistant</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Chat messages container
    st.sidebar.markdown("""
        <div class="chat-container">
            <div class="chat-messages" id="chat-messages">
    """, unsafe_allow_html=True)
    
    # Display welcome message if no chat history
    if len(st.session_state.chat_history) == 0:
        st.sidebar.markdown(
            '<div class="bot-message">👋 Hi there! I\'m the RideWise Assistant. How can I help you today?</div>',
            unsafe_allow_html=True
        )
    
    # Display existing chat messages
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.sidebar.markdown(
                f'<div class="user-message">{message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.sidebar.markdown(
                f'<div class="bot-message">{message["content"]}</div>',
                unsafe_allow_html=True
            )
    
    st.sidebar.markdown('</div></div>', unsafe_allow_html=True)
    
    # Chat input container
    st.sidebar.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    
    # Create a form for the chat input
    with st.sidebar.form(key="chat_form", clear_on_submit=True):
    # Input field for user message
        user_input = st.text_input(
            "Message RideWise Assistant...",
            key="user_message",
            label_visibility="collapsed"
        )
    
        # Send button below the input field
        send_button = st.form_submit_button("Send")

    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Add JavaScript for auto-scrolling to bottom of chat
    st.sidebar.markdown("""
        <script>
            // Function to scroll chat to bottom
            function scrollChatToBottom() {
                const chatMessages = document.getElementById('chat-messages');
                if (chatMessages) {
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            }
            
            // Call scroll function when page loads
            window.onload = scrollChatToBottom;
            
            // Set up a MutationObserver to watch for changes to the chat container
            const observer = new MutationObserver(scrollChatToBottom);
            
            // Start observing the chat container
            const chatContainer = document.getElementById('chat-messages');
            if (chatContainer) {
                observer.observe(chatContainer, { childList: true, subtree: true });
            }
        </script>
    """, unsafe_allow_html=True)
    
    # Process new message
    if send_button and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get response from the unified function
        with st.spinner("Thinking..."):
            response = get_assistant_response(st.session_state.llm_config, user_input)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update UI
        st.rerun()

def main():
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "main"
        
    # Load data and preprocessing (common for all pages)
    data = loadData()
    X_train, X_test, y_train, y_test, le = preprocessing(data)
    
    # Navigation logic
    if st.session_state.page == "compare_models":
        compare_all_models(X_train, X_test, y_train, y_test)
    else:  # Main page
        st.markdown('<div class="gradient-text">RideWise: Predicting Bike Trip Membership Types</div>', unsafe_allow_html=True)
        
        # Show Raw Data section
        if st.checkbox('Show Raw Data'):
            st.subheader("Raw Data Sample:")
            st.write(data.head())
            st.markdown('<hr>', unsafe_allow_html=True)
        
        # Model selection
        choose_model = st.sidebar.selectbox("Choose the ML Model",
            ["NONE", "Decision Tree", "K-Nearest Neighbours", 
             "SVM", "Naive Bayes", "Random Forest", "Logistic Regression"])
        
        # Add a button for model comparison
        if st.sidebar.button("Compare All Models"):
            st.session_state.page = "compare_models"
            st.rerun()
        
        # Individual model handling - no glass boxes
        if choose_model != "NONE":
            st.subheader(f"{choose_model} Model Analysis")
            
            if choose_model == "Decision Tree":
                score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
                st.text("Accuracy of Decision Tree model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()         
                        if st.button("Predict"):
                            pred = tree.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
        
            elif choose_model == "K-Nearest Neighbours":
                score, report, clf = Knn_Classifier(X_train, X_test, y_train, y_test)
                st.text("Accuracy of KNN model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()
                        if st.button("Predict"):
                            pred = clf.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
            
            elif choose_model == "SVM":
                score, report, clf, scaler = svm_classifier(X_train, X_test, y_train, y_test)
                st.text("Accuracy of SVM model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()
                        if st.button("Predict"):
                            user_prediction_data = scaler.transform(user_prediction_data)
                            pred = clf.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
            
            elif choose_model == "Naive Bayes":
                score, report, clf = naive_bayes_classifier(X_train, X_test, y_train, y_test)
                st.text("Accuracy of Naive Bayes model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()
                        if st.button("Predict"):
                            pred = clf.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:st.error(f"Error: {e}. Please enter valid values.")
            
            elif choose_model == "Random Forest":
                score, report, clf = random_forest_classifier(X_train, X_test, y_train, y_test)
                st.text("Accuracy of Random Forest model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()
                        if st.button("Predict"):
                            pred = clf.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
            
            elif choose_model == "Logistic Regression":
                score, report, clf, scaler = logistic_regression_classifier(X_train, X_test, y_train, y_test)
                st.text("Accuracy of Logistic Regression model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()
                        if st.button("Predict"):
                            user_prediction_data = scaler.transform(user_prediction_data)
                            pred = clf.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
        
        # Add map visualization in main page
        st.subheader("Bike Station Map")
        try:
            df = showMap()
            st.map(df)
        except Exception as e:
            st.error(f"Error loading map: {e}")

        # Add visualization selection functionality - missing in the second file
        choose_viz = st.sidebar.selectbox("Choose Visualization",
    ["NONE", "Total number of vehicles from various Starting Points",
     "Total number of vehicles from various End Points",
     "Count of each Member Type"])

# Add visualization code - this is found in the first file but missing in the second
    if choose_viz != "NONE":
      st.subheader(choose_viz)
    if choose_viz == "Total number of vehicles from various Starting Points":
        fig = px.histogram(data['Start station'], x='Start station')
        st.plotly_chart(fig)
    elif choose_viz == "Total number of vehicles from various End Points":
        fig = px.histogram(data['End station'], x='End station')
        st.plotly_chart(fig)
    elif choose_viz == "Count of each Member Type":
        fig = px.histogram(data['Member type'], x='Member type')
        st.plotly_chart(fig)

    # Add back button functionality - this exists in the first file but not the second
    st.sidebar.markdown("---")
    if st.sidebar.button("Back to Main App"):
        st.sidebar.info("Redirecting back to main application...")
        other_app_url = "http://localhost:8501/"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={other_app_url}">', unsafe_allow_html=True)  
        # Display a summary of the application in the main page
        with st.expander("About RideWise"):
            st.markdown("""
            **RideWise** is a machine learning application that analyzes bike sharing system data to predict membership types.
            
            The application uses various machine learning models to classify trips as either:
            - Registered members
            - Casual users
            
            Features used for prediction include:
            - Trip duration
            - Start station
            - End station
            
            You can select different models from the sidebar, compare their performance, and even make predictions with your own input data.
            """)
    
    # Display the chat interface in the sidebar for all pages
    display_chat_interface()

if __name__ == "__main__":
    main()