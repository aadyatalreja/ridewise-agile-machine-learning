import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from backend_qml import (
    loadData, 
    preprocessing, 
    qnn_classifier, 
    qsvm_classifier, 
    qvqc_classifier, 
    qknn_classifier,
    qqcl_classifier,
    compare_qml_models
)
from chatbot_frontend import display_chat_sidebar, display_chat_interface, display_chat_expander, display_chat_tab
from chatbot_backend import setup_llm_assistant, get_assistant_response
# Custom CSS Styling - quantum-inspired design
st.set_page_config(page_title="RideWise: QML Analysis", layout="centered")
display_chat_interface()

# Apply styling with quantum computing theme
st.markdown("""
    <style>
    /* Animated gradient for the title - quantum blue/purple theme */
    .gradient-text {
        font-size: 2.5em;
        font-weight: 800;
        background: linear-gradient(90deg, #3a0ca3 0%, #4361ee 50%, #7209b7 100%);
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

    /* Button styling with gradient */
    div.stButton > button {
        background: linear-gradient(to right, #3a0ca3, #7209b7);
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
            user_prediction_data = np.array([[float(duration), float(start_station), float(end_station)]])
            return user_prediction_data
        return None
    except ValueError:
        st.error("Please enter valid numeric values")
        return None

def showMap():
    try:
        df = pd.read_csv("2010-capitalbikeshare-tripdata.csv")
        # This is a placeholder - in a real implementation, you would have latitude and longitude data
        # For now, we'll create random coordinates centered around Washington DC
        lat_center, lon_center = 38.9072, -77.0369  # Washington DC coordinates
        
        # Create a sample dataframe with fake coordinates
        plotData = pd.DataFrame({
            'lat': np.random.normal(lat_center, 0.02, size=100),
            'lon': np.random.normal(lon_center, 0.02, size=100)
        })
        
        return plotData
    except Exception as e:
        print(f"Error loading map data: {e}")
        # Return some dummy data if the file is not found
        return pd.DataFrame({
            'lat': np.random.normal(38.9072, 0.02, size=20),
            'lon': np.random.normal(-77.0369, 0.02, size=20)
        })

def compare_models_view(X_train, X_test, y_train, y_test):
    st.markdown('<div class="gradient-text">Quantum Machine Learning Model Comparison Dashboard</div>', unsafe_allow_html=True)
    st.write("Comparing the performance of quantum machine learning models for bike trip membership prediction")
    
    # Create a progress bar to show the model training progress
    progress_bar = st.progress(0)
    
    with st.spinner("Running quantum model comparison - this may take a few minutes..."):
        # Get model comparison dataframe
        df_models = compare_qml_models(X_train, X_test, y_train, y_test)
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
        title='Quantum Machine Learning Model Accuracy Comparison',
        text='Accuracy (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify the best performing model
    best_model = df_models.iloc[0]['Model']
    best_accuracy = df_models.iloc[0]['Accuracy (%)']
    
    st.success(f"The best performing quantum model is **{best_model}** with an accuracy of **{best_accuracy:.2f}%**")
    
    # Add quantum advantages explanation
    st.subheader("Understanding Quantum Machine Learning Advantages")
    st.write("""
    Quantum Machine Learning models offer several potential advantages for complex data analysis:
    
    - **Quantum Superposition**: Can process multiple states simultaneously
    - **Quantum Entanglement**: Enables novel correlations between quantum bits
    - **Interference**: Can amplify correct solutions while suppressing wrong ones
    - **Quantum Kernels**: Can access higher-dimensional feature spaces implicitly
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
        st.markdown('<div class="gradient-text">RideWise: Quantum Learning for Bike Trip Classification</div>', unsafe_allow_html=True)
        
        # Show Raw Data section
        if st.checkbox('Show Raw Data'):
            st.subheader("Raw Data Sample:")
            st.write(data.head())
            st.markdown('<hr>', unsafe_allow_html=True)
        
        # Quantum model selection
        choose_model = st.sidebar.selectbox("Choose a Quantum Machine Learning Model",
            ["NONE", "Quantum Neural Network (QNN)", 
             "Quantum Support Vector Machine (QSVM)", 
             "Quantum Variational Quantum Classifier (QVQC)",
             "Quantum K-Nearest Neighbors (QKNN)",
             "Quantum Quantum Clustering (QQCL)"])
        
        # Add a button for model comparison
        if st.sidebar.button("Compare All Quantum Models"):
            st.session_state.page = "compare_models"
            # st.rerun()
        
        # About Quantum Machine Learning section
        with st.sidebar.expander("About Quantum Machine Learning Models"):
            st.write("""
            **Quantum Machine Learning Models** leverage quantum computing principles to enhance 
            classical machine learning tasks. They can potentially offer computational advantages 
            for certain problems and datasets.
            
            The models in this app represent different quantum approaches:
            
            1. **QNN**: Quantum Neural Network with angle embedding and strongly entangling layers
            2. **QSVM**: Quantum Support Vector Machine using quantum kernels
            3. **QVQC**: Quantum Variational Quantum Classifier with parameterized circuits
            4. **QKNN**: Quantum K-Nearest Neighbors with quantum distance metrics
            5. **QQCL**: Quantum Quantum Clustering for unsupervised learning
            """)
        
        # Individual model handling
        if choose_model != "NONE":
            st.subheader(f"{choose_model} Analysis")
            
            with st.spinner(f"Training {choose_model}... This may take a few minutes"):
                if choose_model == "Quantum Neural Network (QNN)":
                    score, report, model = qnn_classifier(X_train, X_test, y_train, y_test)
                    
                    # Create two columns for metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Accuracy", f"{score:.2f}%")
                    with col2:
                        st.metric("Model Type", "QNN")
                    
                    # Show classification report
                    with st.expander("Classification Report Details"):
                        st.text("Classification Report:")
                        st.text(report)
                    
                    # Information about model
                    st.info("""
                    **Quantum Neural Network (QNN)** uses quantum circuits with trainable parameters to perform machine learning tasks.
                    It embeds classical data using angle embedding and processes it through strongly entangling layers of quantum operations.
                    The model can potentially capture complex patterns through quantum superposition and entanglement.
                    """)
            
                elif choose_model == "Quantum Support Vector Machine (QSVM)":
                    score, report, model = qsvm_classifier(X_train, X_test, y_train, y_test)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Accuracy", f"{score:.2f}%")
                    with col2:
                        st.metric("Model Type", "QSVM")
                    
                    with st.expander("Classification Report Details"):
                        st.text("Classification Report:")
                        st.text(report)
                    
                    st.info("""
                    **Quantum Support Vector Machine (QSVM)** leverages quantum computing to calculate kernel functions.
                    It maps classical data to quantum feature space using a quantum feature map with Hadamard gates and entangling operations.
                    This potentially allows access to higher-dimensional feature spaces that would be intractable classically.
                    """)
            
                elif choose_model == "Quantum Variational Quantum Classifier (QVQC)":
                    score, report, model = qvqc_classifier(X_train, X_test, y_train, y_test)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Accuracy", f"{score:.2f}%")
                    with col2:
                        st.metric("Model Type", "QVQC")
                    
                    with st.expander("Classification Report Details"):
                        st.text("Classification Report:")
                        st.text(report)
                    
                    st.info("""
                    **Quantum Variational Quantum Classifier (QVQC)** uses parameterized quantum circuits as a machine learning model.
                    It encodes data using rotation gates and applies variational layers with rotation and entangling gates.
                    Parameters are optimized classically to minimize a cost function computed on the quantum device.
                    """)
                    
                elif choose_model == "Quantum K-Nearest Neighbors (QKNN)":
                    score, report, model = qknn_classifier(X_train, X_test, y_train, y_test)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Accuracy", f"{score:.2f}%")
                    with col2:
                        st.metric("Model Type", "QKNN")
                    
                    with st.expander("Classification Report Details"):
                        st.text("Classification Report:")
                        st.text(report)
                    
                    st.info("""
                    **Quantum K-Nearest Neighbors (QKNN)** is a quantum version of the classical KNN algorithm.
                    It uses quantum computing techniques to measure distance/similarity between data points.
                    This model can leverage quantum interference to potentially offer speedups in distance calculations.
                    """)
                    
                elif choose_model == "Quantum Quantum Clustering (QQCL)":
                    score, report, model = qqcl_classifier(X_train, X_test, y_train, y_test)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Accuracy", f"{score:.2f}%")
                    with col2:
                        st.metric("Model Type", "QQCL")
                    
                    with st.expander("Classification Report Details"):
                        st.text("Classification Report:")
                        st.text(report)
                    
                    st.info("""
                    **Quantum Quantum Clustering (QQCL)** applies quantum principles to clustering tasks.
                    It uses quantum distance metrics to assign data points to clusters and refine centroids.
                    The algorithm can potentially identify complex cluster structures through quantum state evolution.
                    """)
            
                # User prediction interface
                st.subheader("Try a Prediction")
                try:
                    st.info("Enter values to predict membership type using the trained quantum model")
                    user_prediction_data = accept_user_data_input()        
                    if user_prediction_data is not None and st.button("Predict with Quantum Model"):
                        with st.spinner("Running quantum prediction..."):
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
    if st.sidebar.button("Back to Main App"):
        st.switch_page("pages/home_page.py")

if __name__ == "__main__":
    main()