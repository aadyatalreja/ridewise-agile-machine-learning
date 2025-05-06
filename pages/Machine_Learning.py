import streamlit as st
import pandas as pd
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

# Import our custom chatbot modules
from chatbot_frontend import display_chat_sidebar, display_chat_interface, display_chat_expander, display_chat_tab
from chatbot_backend import setup_llm_assistant, get_assistant_response

# Apply custom styling
st.set_page_config(page_title="RideWise ML Analysis", layout="centered")
# Display the chat interface in the sidebar for all pages
display_chat_interface()
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
    
    /* Message fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
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
        # st.rerun()

def main():
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "main"
        
    # Initialize LLM assistant on first load
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config = setup_llm_assistant()
        
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
            # st.rerun()

        with st.sidebar.expander("About Machine Learning Models"):
            st.write("""
                 **Machine Learning Models** are algorithms that learn patterns from data to make predictions or decisions 
                 without being explicitly programmed for each task. These models rely on statistical techniques and vary 
                in complexity and interpretability.

                The models in this app include classic machine learning approaches:

                1. **Linear Regression**: Models relationships using linear equations; ideal for continuous outcomes
                2. **Logistic Regression**: Used for binary classification tasks based on probability estimation
                3. **Decision Tree**: A tree-based model that splits data based on feature thresholds
                4. **Random Forest**: An ensemble of decision trees that improves accuracy and reduces overfitting
                5. **Support Vector Machine (SVM)**: Finds the optimal boundary to separate classes in high-dimensional space
            """)
        
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
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
            
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
        st.subheader("Bike Trip Start Locations")
        try:
            plotData = showMap()
            st.map(plotData, zoom=14)
        except Exception as e:
            st.warning(f"Could not load map data. Make sure the CSV file is in the correct location.")
            st.error(f"Error details: {e}")

        # Add visualization selection functionality
        choose_viz = st.sidebar.selectbox("Choose Visualization",
            ["NONE", "Total number of vehicles from various Starting Points",
             "Total number of vehicles from various End Points",
             "Count of each Member Type"])

        # Add visualization code
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

        # Add back button functionality
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