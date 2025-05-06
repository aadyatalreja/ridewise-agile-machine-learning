import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_preprocess_data():
    data = loadData()
    X_train, X_test, y_train, y_test, le = preprocessing(data)
    return data, X_train, X_test, y_train, y_test, le

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
        # Get model comparison dataframe
        df_models = compare_all_qnn_models(X_train, X_test, y_train, y_test)
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
    
    st.success(f"The best performing quantum model is **{best_model}** with an accuracy of **{best_accuracy:.2f}%**")
    
    # Add quantum advantage explanation
    st.subheader("Understanding Quantum Advantage")
    st.write("""
    Quantum Neural Networks can offer potential advantages over classical models:
    
    - **Quantum Superposition**: QNNs can process multiple states simultaneously
    - **Quantum Entanglement**: Creates correlations that classical models can't achieve
    - **Feature Space**: Quantum models can access higher-dimensional feature spaces
    
    However, current quantum models are still in their early stages and may not always outperform classical models.
    """)
    
    # Add a button to return to the main page
    if st.button("Return to Main Page"):
        st.session_state.page = "main"
        # st.rerun()

def main():
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "main"
        
    # Load data and preprocessing (common for all pages)
    data, X_train, X_test, y_train, y_test, le = load_and_preprocess_data()
    
    # Navigation logic
    if st.session_state.page == "compare_models":
        compare_models_view(X_train, X_test, y_train, y_test)
    else:  # Main page
        st.markdown('<div class="gradient-text">RideWise: Quantum Neural Networks for Bike Trip Classification</div>', unsafe_allow_html=True)
        
        # Show Raw Data section
        if st.checkbox('Show Raw Data'):
            st.subheader("Raw Data Sample:")
            st.write(data.head())
            st.markdown('<hr>', unsafe_allow_html=True)
        
        # Quantum model selection
        choose_model = st.sidebar.selectbox("Choose a Quantum Neural Network",
            ["NONE", "Variational Quantum Classifier (VQC)", 
             "Quantum Convolutional Neural Network (QCNN)", 
             "Quantum Feedforward Neural Network (QFNN)"])
        
        # Add a button for model comparison
        if st.sidebar.button("Compare All Quantum Models"):
            st.session_state.page = "compare_models"
            # st.rerun()
        
        # About Quantum Computing section
        with st.sidebar.expander("About Quantum Neural Networks"):
            st.write("""
            **Quantum Neural Networks (QNNs)** combine quantum computing with neural network concepts. 
            They leverage quantum phenomena like superposition and entanglement to potentially solve 
            problems that classical neural networks struggle with.
            
            The three models in this app represent different approaches to quantum machine learning:
            
            1. **VQC**: Variational circuits for classification tasks
            2. **QCNN**: Quantum version of convolutional neural networks
            3. **QFNN**: Quantum implementation of feedforward architecture
            """)
        
        # Individual model handling
        if choose_model != "NONE":
            st.subheader(f"{choose_model} Analysis")
            
            with st.spinner(f"Training {choose_model}... This may take a few minutes"):
                if choose_model == "Variational Quantum Classifier (VQC)":
                    score, report, model = vqc_classifier(X_train, X_test, y_train, y_test)
                    
                    # Create two columns for metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Accuracy", f"{score:.2f}%")
                    with col2:
                        st.metric("Model Type", "Variational Circuit")
                    
                    # Show classification report
                    with st.expander("Classification Report Details"):
                        st.text("Classification Report:")
                        st.text(report)
                    
                    # Information about model
                    st.info("""
                    **Variational Quantum Classifier (VQC)** uses parameterized quantum circuits to create a 
                    quantum version of a neural network. This model embeds data into quantum states and applies 
                    strongly entangling layers to exploit quantum properties for classification.
                    """)
            
                elif choose_model == "Quantum Convolutional Neural Network (QCNN)":
                    score, report, model = qcnn_classifier(X_train, X_test, y_train, y_test)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Accuracy", f"{score:.2f}%")
                    with col2:
                        st.metric("Model Type", "Quantum CNN")
                    
                    with st.expander("Classification Report Details"):
                        st.text("Classification Report:")
                        st.text(report)
                    
                    st.info("""
                    **Quantum Convolutional Neural Network (QCNN)** is the quantum analog of classical CNNs. 
                    It applies quantum convolutional layers followed by pooling operations, 
                    reducing the number of active qubits and extracting hierarchical features.
                    """)
            
                elif choose_model == "Quantum Feedforward Neural Network (QFNN)":
                    score, report, model = qfnn_classifier(X_train, X_test, y_train, y_test)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Accuracy", f"{score:.2f}%")
                    with col2:
                        st.metric("Model Type", "Quantum Feedforward")
                    
                    with st.expander("Classification Report Details"):
                        st.text("Classification Report:")
                        st.text(report)
                    
                    st.info("""
                    **Quantum Feedforward Neural Network (QFNN)** mimics classical feedforward neural networks 
                    with input, hidden, and output layers, but uses quantum operations instead of classical 
                    activation functions. This allows for potentially more complex data transformations.
                    """)
            
                # User prediction interface
                st.subheader("Try a Prediction")
                try:
                    st.info("Enter values to predict membership type using the trained quantum model")
                    user_prediction_data = accept_user_data_input()        
                    if user_prediction_data is not None and st.button("Predict with Quantum Model"):
                        with st.spinner("Running quantum prediction..."):
                            if choose_model == "Variational Quantum Classifier (VQC)":
                                pred = model.predict(user_prediction_data)
                            elif choose_model == "Quantum Convolutional Neural Network (QCNN)":
                                pred = model.predict(user_prediction_data)
                            elif choose_model == "Quantum Feedforward Neural Network (QFNN)":
                                pred = model.predict(user_prediction_data)
                            
                            # Display prediction with appropriate styling
                            membership_type = le.inverse_transform(pred)[0]
                            if membership_type == "Member":
                                st.success(f"Predicted Membership Type: {membership_type}")
                            else:
                                st.warning(f"Predicted Membership Type: {membership_type}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            
            # Add a separator
            st.markdown('<hr>', unsafe_allow_html=True)
        
        # Map and Visualization
        st.subheader("Bike Trip Start Locations")
        try:
            plotData = showMap()
            st.map(plotData, zoom=14)
        except Exception as e:
            st.warning(f"Could not load map data. Make sure the CSV file is in the correct location.")
            st.error(f"Error details: {e}")
    
        choose_viz = st.sidebar.selectbox("Choose Visualization",
            ["NONE", "Total number of vehicles from various Starting Points",
             "Total number of vehicles from various End Points",
             "Count of each Member Type"])
        
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
            
    # Add back button
    st.sidebar.markdown("---")
    with st.sidebar.expander("About RideWise"):
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
    if st.sidebar.button("Back to Home Page"):
        st.switch_page("pages/home_page.py")

if __name__ == "__main__":
    main()