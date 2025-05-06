import streamlit as st
import google.generativeai as genai

def setup_llm_assistant():
    """
    Sets up an LLM assistant with multiple provider options.
    Tries Gemini first, then OpenAI, then falls back to local.
    Returns a configuration dictionary with provider and model information.
    """
    # First try Gemini
    gemini_model = setup_gemini()
    if gemini_model:
        return {"provider": "gemini", "model": gemini_model}
    
    # If Gemini fails, try OpenAI
    # try:
    #     OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    #     if OPENAI_API_KEY:
    #         openai.api_key = OPENAI_API_KEY
    #         # Test the OpenAI connection
    #         test_response = openai.chat.completions.create(
    #             model="gpt-3.5-turbo",
    #             messages=[{"role": "user", "content": "Hello"}],
    #             max_tokens=10
    #         )
    #         if test_response:
    #             st.sidebar.success("Connected to OpenAI (fallback mode)")
    #             return {"provider": "openai", "model": "gpt-3.5-turbo"}
    # except Exception as e:
    #     st.sidebar.warning(f"OpenAI fallback failed: {str(e)}")
    
    # If both fail, use local fallback
    st.sidebar.warning("Using basic local assistant (all LLM services unavailable)")
    return {"provider": "local", "model": None}

def setup_gemini():
    """
    Sets up the Gemini API with fallback options for different models.
    Returns a working model or None if all attempts fail.
    """
    try:
        # In production, use st.secrets to manage API keys
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            st.sidebar.warning("Gemini API key not found in secrets. Assistant features are limited.")
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
        st.sidebar.error("All Gemini model options failed. Assistant features are limited.")
        return None
        
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Gemini API: {str(e)}")
        return None

def get_gemini_response(model, user_input):
    """
    Get response from Gemini with improved error handling
    """
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

def local_fallback_assistant(user_input):
    """
    Provides basic responses when LLM APIs are unavailable.
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

def get_assistant_response(llm_config, user_input):
    """
    Unified function to get responses from any available LLM provider
    """
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
    
    # elif provider == "openai":
        # try:
        #     response = openai.chat.completions.create(
        #         model=llm_config["model"],
        #         messages=[
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": user_input}
        #         ],
        #         max_tokens=300,
        #         temperature=0.7
        #     )
        #     return response.choices[0].message.content
        # except Exception as e:
        #     return f"OpenAI Assistant Error: {str(e)}"
    
    else:  # local fallback
        return local_fallback_assistant(user_input)

# Function to initialize session state for chat history
def initialize_chat_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config = setup_llm_assistant()