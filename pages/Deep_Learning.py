import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from backend_dl import (
    loadData, 
    preprocessing, 
    cnn_classifier, 
    lstm_classifier, 
    mlp_classifier, 
    bilstm_classifier,
    wide_deep_classifier,
    compare_dl_models
)
from chatbot_frontend import display_chat_sidebar, display_chat_interface, display_chat_expander, display_chat_tab
from chatbot_backend import setup_llm_assistant, get_assistant_response
# Custom CSS Styling - deep learning-inspired design
st.set_page_config(page_title="RideWise: DL Analysis", layout="centered")
# Display the chat interface in the sidebar for all pages
display_chat_interface()
# Apply styling with deep learning theme
st.markdown("""
    <style>
    /* Animated gradient for the title - deep learning blue/purple theme */
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

    /* Override tab underline to deep learning blue */
    div[data-baseweb="tab"] button[aria-selected="true"] {
        border-bottom: 3px solid #4361ee;
        color: #4361ee;
    }

    /* Button styling with gradient */
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
    st.markdown('<div class="gradient-text">Deep Learning Model Comparison Dashboard</div>', unsafe_allow_html=True)
    st.write("Comparing the performance of deep learning models for bike trip membership prediction")
    
    # Create a progress bar to show the model training progress
    progress_bar = st.progress(0)
    
    # Only run model comparison if results aren't already in session state
    if 'model_comparison_results' not in st.session_state:
        with st.spinner("Running model comparison - this may take a few minutes..."):
            # Get model comparison dataframe
            st.session_state.model_comparison_results = compare_dl_models(X_train, X_test, y_train, y_test)
            progress_bar.progress(100)
    else:
        # If results already exist, just show completed progress bar
        progress_bar.progress(100)
    
    # Get results from session state
    df_models = st.session_state.model_comparison_results
    
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
        title='Deep Learning Model Accuracy Comparison',
        text='Accuracy (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify the best performing model(s)
    best_accuracy = df_models['Accuracy (%)'].max()
    best_models = df_models[df_models['Accuracy (%)'] == best_accuracy]['Model'].tolist()
    
    if len(best_models) == 1:
        st.success(f"The best performing deep learning model is *{best_models[0]}* with an accuracy of *{best_accuracy:.2f}%*")
    else:
        best_models_str = ", ".join([f"{model}" for model in best_models])
        st.success(f"The best performing deep learning models are {best_models_str}, all with an accuracy of *{best_accuracy:.2f}%*")
    
    # Add deep learning advantages explanation
    st.subheader("Understanding Deep Learning Advantages")
    st.write("""
    Deep Learning models offer several advantages for complex data analysis:
    
    - *Feature Learning*: Deep models automatically learn hierarchical features from data
    - *Flexibility*: Different architectures can handle various data types and patterns
    - *Non-linearity*: Can model highly complex non-linear relationships
    - *Scalability*: Performance typically improves with more data and computing resources
    """)
    
    # Fix: Return to Main Page button - now simply changes the session state without rerunning
    if st.button("Return to Main Page"):
        st.session_state.page = "main"
        st.rerun()  # Use st.rerun() for newer Streamlit versions

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
        st.markdown('<div class="gradient-text">RideWise: Deep Learning for Bike Trip Classification</div>', unsafe_allow_html=True)
        
        # Show Raw Data section
        if st.checkbox('Show Raw Data'):
            st.subheader("Raw Data Sample:")
            st.write(data.head())
            st.markdown('<hr>', unsafe_allow_html=True)
        
        # Deep Learning model selection
        choose_model = st.sidebar.selectbox("Choose a Deep Learning Model",
            ["NONE", "Convolutional Neural Network (CNN)", 
             "Long Short-Term Memory (LSTM)", 
             "Multi-Layer Perceptron (MLP)",
             "Bidirectional LSTM (BiLSTM)",
             "Wide & Deep"])
        
        # Add a button for model comparison
        if st.sidebar.button("Compare All Deep Learning Models"):
            st.session_state.page = "compare_models"
            st.rerun()  # Use st.rerun() for newer Streamlit versions
        
        # About Deep Learning section
        with st.sidebar.expander("About Deep Learning Models"):
            st.write("""
            *Deep Learning Models* are advanced neural networks with multiple hidden layers that can learn 
            complex patterns in data. They excel at feature extraction and can model highly non-linear relationships.
            
            The models in this app represent different deep learning architectures:
            
            1. *CNN*: Excels at spatial pattern recognition, even in 1D data
            2. *LSTM*: Specialized for sequential data and temporal dependencies
            3. *MLP*: Traditional fully-connected neural network architecture
            4. *BiLSTM*: Bidirectional LSTM that can learn from both past and future context
            5. *Wide & Deep*: Combines memorization and generalization abilities
            """)
        
        # Individual model handling
        if choose_model != "NONE":
            st.subheader(f"{choose_model} Analysis")
            
            # Use session state to store model results to avoid retraining
            model_key = choose_model.replace(" ", "_").lower()
            
            if model_key not in st.session_state:
                with st.spinner(f"Training {choose_model}... This may take a few minutes"):
                    if choose_model == "Convolutional Neural Network (CNN)":
                        score, report, model = cnn_classifier(X_train, X_test, y_train, y_test)
                    elif choose_model == "Long Short-Term Memory (LSTM)":
                        score, report, model = lstm_classifier(X_train, X_test, y_train, y_test)
                    elif choose_model == "Multi-Layer Perceptron (MLP)":
                        score, report, model = mlp_classifier(X_train, X_test, y_train, y_test)
                    elif choose_model == "Bidirectional LSTM (BiLSTM)":
                        score, report, model = bilstm_classifier(X_train, X_test, y_train, y_test)
                    elif choose_model == "Wide & Deep":
                        score, report, model = wide_deep_classifier(X_train, X_test, y_train, y_test)
                    
                    # Store results in session state
                    st.session_state[model_key] = {
                        'score': score,
                        'report': report,
                        'model': model
                    }
            
            # Retrieve model results from session state
            score = st.session_state[model_key]['score']
            report = st.session_state[model_key]['report']
            model = st.session_state[model_key]['model']
            
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Accuracy", f"{score:.2f}%")
            with col2:
                st.metric("Model Type", choose_model.split(" (")[0])
            
            # Show classification report
            with st.expander("Classification Report Details"):
                st.text("Classification Report:")
                st.text(report)
            
            # Information about the model based on model type
            if choose_model == "Convolutional Neural Network (CNN)":
                st.info("""
                *Convolutional Neural Network (CNN)* applies convolutional filters to extract local patterns 
                in data. While traditionally used for images, CNNs can also detect patterns in tabular or 
                temporal data when the features are arranged meaningfully.
                """)
            elif choose_model == "Long Short-Term Memory (LSTM)":
                st.info("""
                *Long Short-Term Memory (LSTM)* networks are a type of recurrent neural network (RNN) 
                specifically designed to handle long-term dependencies. They use memory cells to store, 
                forget, and access information over long sequences.
                """)
            elif choose_model == "Multi-Layer Perceptron (MLP)":
                st.info("""
                *Multi-Layer Perceptron (MLP)* is a classic feedforward neural network with multiple hidden layers.
                It connects each neuron in one layer to every neuron in the next layer, allowing it to learn
                complex non-linear relationships between features.
                """)
            elif choose_model == "Bidirectional LSTM (BiLSTM)":
                st.info("""
                *Bidirectional LSTM (BiLSTM)* processes data in both forward and backward directions,
                allowing the network to learn from both past and future context. This bidirectional approach
                often captures more comprehensive patterns in sequential data.
                """)
            elif choose_model == "Wide & Deep":
                st.info("""
                *Wide & Deep* architecture combines two components: a wide linear model for memorization 
                and a deep neural network for generalization. This combination allows it to learn both 
                broad patterns and specific feature interactions simultaneously.
                """)
            
            # User prediction interface
            st.subheader("Try a Prediction")
            try:
                st.info("Enter values to predict membership type using the trained deep learning model")
                user_prediction_data = accept_user_data_input()        
                if user_prediction_data is not None and st.button("Predict with Deep Learning Model"):
                    with st.spinner("Running prediction..."):
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
    # Display a summary of the application in the main page
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