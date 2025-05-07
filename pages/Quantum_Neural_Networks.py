import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend_qnn import (
    loadData, 
    preprocessing, 
    vqc_classifier, 
    qcnn_classifier, 
    qfnn_classifier, 
    compare_all_qnn_models, 
    showMap, 
    accept_user_data
)
from chatbot_frontend import display_chat_sidebar, display_chat_interface, display_chat_expander, display_chat_tab
from chatbot_backend import setup_llm_assistant, get_assistant_response

# Custom CSS Styling - quantum-inspired design
st.set_page_config(page_title="RideWise: QNN Analysis", layout="centered")
# Display the chat interface in the sidebar for all pages
display_chat_interface()
# Apply styling with quantum-inspired theme
st.markdown("""
    <style>
    /* Animated gradient for the title - quantum blue/purple theme */
    .gradient-text {
        font-size: 2.5em;
        font-weight: 800;
        background: linear-gradient(90deg, #3a0ca3 0%, #4361ee 50%, #4cc9f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-flow 3s infinite alternate;
    }

    @keyframes gradient-flow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    /* Standard section styling */
    .section-container {
        padding: 1.5rem 0;
        margin: 1rem 0;
        border-bottom: 1px solid #f0f0f0;
    }

    /* Override tab underline to quantum blue */
    div[data-baseweb="tab"] button[aria-selected="true"] {
        border-bottom: 3px solid #4361ee;
        color: #4361ee;
    }

    /* Button styling with quantum gradient */
    div.stButton > button {
        background: linear-gradient(to right, #3a0ca3, #4361ee);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75em 1em;
        width: 100%;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: rgba(67, 97, 238, 0.1);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Remove unwanted top empty box */
    section[data-testid="stTabs"] > div:first-child {
        display: none !important;
    }
    
    /* Message fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_preprocess_data():
    data = loadData()
    X_train, X_test, y_train, y_test, le = preprocessing(data)
    return data, X_train, X_test, y_train, y_test, le

# Cache model training functions to avoid retraining when changing pages
@st.cache_resource
def train_vqc_model(X_train, X_test, y_train, y_test):
    return vqc_classifier(X_train, X_test, y_train, y_test)

@st.cache_resource
def train_qcnn_model(X_train, X_test, y_train, y_test):
    return qcnn_classifier(X_train, X_test, y_train, y_test)

@st.cache_resource
def train_qfnn_model(X_train, X_test, y_train, y_test):
    return qfnn_classifier(X_train, X_test, y_train, y_test)

@st.cache_resource
def get_model_comparison(X_train, X_test, y_train, y_test):
    return compare_all_qnn_models(X_train, X_test, y_train, y_test)

def accept_user_data_input():
    duration = st.text_input("Enter the Duration (seconds): ")
    start_station = st.text_input("Enter the start station number: ")
    end_station = st.text_input("Enter the end station number: ")
    
    try:
        if duration and start_station and end_station:
            user_prediction_data = accept_user_data(float(duration), float(start_station), float(end_station))
            return user_prediction_data
        return None
    except ValueError:
        st.error("Please enter valid numeric values")
        return None

def compare_models_view(X_train, X_test, y_train, y_test):
    st.markdown('<div class="gradient-text">Quantum Neural Network Model Comparison Dashboard</div>', unsafe_allow_html=True)
    st.write("Comparing the performance of quantum neural network models for bike trip membership prediction")
    
    # Create a progress bar to show the model training progress
    progress_bar = st.progress(0)
    
    with st.spinner("Running model comparison - this may take a few minutes..."):
        # Get model comparison dataframe using the cached function
        df_models = get_model_comparison(X_train, X_test, y_train, y_test)
        progress_bar.progress(100)
    
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
        title='Quantum Model Accuracy Comparison',
        text='Accuracy (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify the best performing model
    best_model = df_models.iloc[0]['Model']
    best_accuracy = df_models.iloc[0]['Accuracy (%)']
    
    st.success(f"The best performing quantum model is *{best_model}* with an accuracy of *{best_accuracy:.2f}%*")
    
    # Add quantum advantage explanation
    st.subheader("Understanding Quantum Advantage")
    st.write("""
    Quantum Neural Networks can offer potential advantages over classical models:
    
    - *Quantum Superposition*: QNNs can process multiple states simultaneously
    - *Quantum Entanglement*: Creates correlations that classical models can't achieve
    - *Feature Space*: Quantum models can access higher-dimensional feature spaces
    
    However, current quantum models are still in their early stages and may not always outperform classical models.
    """)
    # Add a button to return to the main page
    if st.button("Return to Main Page"):
        st.session_state.page = "main"
        # st.rerun()

def create_sidebar_elements():
    """Create sidebar elements in the specified order"""
    # 1. Choose a QNN Model dropdown
    choose_model = st.sidebar.selectbox(
        "Choose a Quantum Neural Network Model",
        ("None", 
         "Variational Quantum Classifier (VQC)", 
         "Quantum Convolutional Neural Network (QCNN)",
         "Quantum Feedforward Neural Network (QFNN)")
    )
    
    # 2. Compare All Models button
    compare_models_button = st.sidebar.button("Compare All Models")
    
    # 3. About QNN expander
    with st.sidebar.expander("About Quantum Neural Networks"):
        st.write("""
        *Quantum Neural Networks (QNNs)* are machine learning models that leverage quantum computing principles 
        to enhance pattern recognition and prediction capabilities.

        The QNN models in this application include:

        1. *Variational Quantum Classifier (VQC)*: Uses parameterized quantum circuits for classification
        2. *Quantum Convolutional Neural Network (QCNN)*: Applies quantum convolution operations
        3. *Quantum Feedforward Neural Network (QFNN)*: Quantum implementation of classical feedforward networks
        """)
    
    # 4. Choose Visualization dropdown
    choose_viz = st.sidebar.selectbox("Choose Visualization",
        ["NONE", "Total number of vehicles from various Starting Points",
         "Total number of vehicles from various End Points",
         "Count of each Member Type"])
    
    # Add a separator
    st.sidebar.markdown("---")
    
    # 5. About RideWise expander
    with st.sidebar.expander("About RideWise"):
        st.markdown("""
        *RideWise* is a machine learning application that analyzes bike sharing system data to predict membership types.
        
        The application uses various machine learning models to classify trips as either:
        - Registered members
        - Casual users
        
        Features used for prediction include:
        - Trip duration
        - Start station
        - End station
        
        You can select different models from the sidebar, compare their performance, and even make predictions with your own input data.
        """)
    
    # 6. Back to Home Page button
    back_button = st.sidebar.button("Back to Home Page")
    
    
    
    return choose_model, compare_models_button, choose_viz, back_button

def main_view(data, X_train, X_test, y_train, y_test, le, choose_model, choose_viz):
    """Display the main page view"""
    st.markdown('<div class="gradient-text">RideWise: Quantum Neural Networks for Bike Trip Classification</div>', unsafe_allow_html=True)
    
    # Show Raw Data section
    if st.checkbox('Show Raw Data'):
        st.subheader("Raw Data Sample:")
        st.write(data.head())
        st.markdown('<hr>', unsafe_allow_html=True)
    
    # Display model details based on selection
    if choose_model == "None":
        st.subheader("Select a Quantum Neural Network Model")
        st.write("Please select a model from the sidebar to view its details and performance.")
        
    elif choose_model == "Variational Quantum Classifier (VQC)":
        st.subheader("Variational Quantum Classifier (VQC)")
        
        with st.spinner("Training VQC model..."):
            score, report, model = train_vqc_model(X_train, X_test, y_train, y_test)
        
        st.metric("Model Accuracy", f"{score:.2f}%")
        
        st.subheader("Classification Report")
        st.text(report)
        
        st.subheader("Model Architecture")
        st.write("""
        The VQC model uses a parameterized quantum circuit as a classifier:
        
        1. *Data Encoding*: Features are embedded into quantum states
        2. *Quantum Processing*: Parameterized rotation gates and entanglement operations
        3. *Measurement*: Qubit states are measured to determine classification outcome
        """)
        
    elif choose_model == "Quantum Convolutional Neural Network (QCNN)":
        st.subheader("Quantum Convolutional Neural Network (QCNN)")
        
        with st.spinner("Training QCNN model..."):
            score, report, model = train_qcnn_model(X_train, X_test, y_train, y_test)
        
        st.metric("Model Accuracy", f"{score:.2f}%")
        
        st.subheader("Classification Report")
        st.text(report)
        
        st.subheader("Model Architecture")
        st.write("""
        The QCNN model applies convolution and pooling in quantum space:
        
        1. *Quantum Convolution*: Local quantum operations on adjacent qubits
        2. *Quantum Pooling*: Information compression via controlled operations
        3. *Fully Connected Layer*: Final quantum rotations for classification
        """)
        
    elif choose_model == "Quantum Feedforward Neural Network (QFNN)":
        st.subheader("Quantum Feedforward Neural Network (QFNN)")
        
        with st.spinner("Training QFNN model..."):
            score, report, model = train_qfnn_model(X_train, X_test, y_train, y_test)
        
        st.metric("Model Accuracy", f"{score:.2f}%")
        
        st.subheader("Classification Report")
        st.text(report)
        
        st.subheader("Model Architecture")
        st.write("""
        The QFNN model implements a quantum version of a classical neural network:
        
        1. *Input Layer*: Initial quantum rotations based on input features
        2. *Hidden Layers*: Multiple layers of parameterized quantum gates
        3. *Output Layer*: Final rotations for classification results
        """)
    
    # Only show prediction section if a model is selected
    if choose_model != "None":
        # User input for prediction
        st.subheader("Make a Prediction")
        user_data = accept_user_data_input()
        
        if user_data is not None:
            if choose_model == "Variational Quantum Classifier (VQC)":
                _, _, model = train_vqc_model(X_train, X_test, y_train, y_test)
            elif choose_model == "Quantum Convolutional Neural Network (QCNN)":
                _, _, model = train_qcnn_model(X_train, X_test, y_train, y_test)
            else:  # QFNN
                _, _, model = train_qfnn_model(X_train, X_test, y_train, y_test)
                
            prediction = model.predict(user_data)
            membership_type = le.inverse_transform(prediction)[0]
            
            st.success(f"Predicted Membership Type: *{membership_type}*")
            
            # Show confidence based on distance from decision boundary
            st.write("Model confidence visualization:")
            confidence = abs(0.5 - prediction[0])
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Prediction Confidence"},
                gauge = {
                    'axis': {'range': [None, 50]},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': confidence * 100}}))
            
            st.plotly_chart(fig)
    
    # Show map of bike stations
    st.subheader("Bike Station Map")
    map_data = showMap()
    st.map(map_data)
    
    # Add visualization based on selection
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



def main():
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "main"
    
    # Initialize LLM assistant on first load
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config = setup_llm_assistant()
        
    # Load data and preprocessing (common for all pages)
    data, X_train, X_test, y_train, y_test, le = load_and_preprocess_data()
    
    # Create sidebar elements
    choose_model, compare_models_button, choose_viz, back_button = create_sidebar_elements()
    
    # Handle button clicks
    if compare_models_button:
        st.session_state.page = "compare_models"
        st.rerun()
        
    if back_button:
        st.switch_page("pages/home_page.py")
    
    # Navigation logic
    if st.session_state.page == "compare_models":
        compare_models_view(X_train, X_test, y_train, y_test)
    else:  # Main page
        main_view(data, X_train, X_test, y_train, y_test, le, choose_model, choose_viz)

if __name__ == "__main__":
    main()