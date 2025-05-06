import streamlit as st
from chatbot_backend import initialize_chat_state, get_assistant_response

def add_chat_styling():
    """
    Add CSS styling for the chat interface with a modern RiseWise-style gradient for both user and bot messages
    """
    st.markdown("""
        <style>
        /* User message style with gradient and slight glassmorphism */
        .user-message {
            background: linear-gradient(135deg, rgba(255, 75, 145, 0.7), rgba(255, 69, 162, 0.7));
            backdrop-filter: blur(6px);
            color: #ffffff;
            border-radius: 18px 18px 4px 18px;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
            max-width: 80%;
            margin-left: auto;
            word-wrap: break-word;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
            animation: fadeIn 0.3s ease-out;
        }

        /* Bot message style - RiseWise Assistant theme */
        .bot-message {
            background: linear-gradient(135deg, #ff4b91 100%, #a100f2 0%);
            color: white;
            border-radius: 18px 18px 18px 0;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
            max-width: 80%;
            margin-right: auto;
            word-wrap: break-word;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            animation: fadeIn 0.3s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .chat-header {
            background: linear-gradient(90deg, #ff4b91 0%, #a100f2 100%);
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

        .chat-messages {
            padding: 1rem;
            overflow-y: auto;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            max-height: 400px;
        }

        .toggle-chat-btn {
            padding: 0.5rem 1rem;
            background: linear-gradient(90deg, #ff4b91 0%, #a100f2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-weight: 600;
            text-align: center;
            margin: 0.5rem 0;
            width: 100%;
            transition: all 0.3s ease;
        }

        .toggle-chat-btn:hover {
            background: linear-gradient(90deg, #e63e81 0%, #8b00d1 100%);
        }
        </style>

        <script>
        function scrollChatToBottom() {
            const chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }

        window.onload = scrollChatToBottom;

        const observer = new MutationObserver(scrollChatToBottom);
        const chatContainer = document.getElementById('chat-messages');
        if (chatContainer) {
            observer.observe(chatContainer, { childList: true, subtree: true });
        }
        </script>
    """, unsafe_allow_html=True)


def display_chat_interface(container=None):
    """
    Display the chat interface in the given container.
    If no container is provided, it will display in the current Streamlit container.
    
    Args:
        container: Optional Streamlit container to place the chat interface in
    """
    # Initialize chat state
    initialize_chat_state()
    
    # Apply styling
    add_chat_styling()
    
    # Create a context manager for the appropriate container
    if container is None:
        ctx = st
    else:
        ctx = container
    
    # Option to toggle chat visibility
    if 'chat_visible' not in st.session_state:
        st.session_state.chat_visible = False
    
    # Chat toggle button
    if ctx.button("💬 Toggle RideWise Assistant", key="toggle_chat_btn"):
        st.session_state.chat_visible = not st.session_state.chat_visible
    
    # Only display chat if visible
    if st.session_state.chat_visible:
        # Chat header
        ctx.markdown("""
            <div class="chat-header">
                <div class="chat-header-title">
                    <span>🤖</span>
                    <span>RideWise Assistant</span>
                </div>
            </div>
            <div class="chat-container">
                <div class="chat-messages" id="chat-messages">
        """, unsafe_allow_html=True)
        
        # Display welcome message if no chat history
        if len(st.session_state.chat_history) == 0:
            ctx.markdown(
                '<div class="bot-message">👋 Hi there! I\'m the RideWise Assistant. How can I help you today?</div>',
                unsafe_allow_html=True
            )
        
        # Display existing chat messages
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                ctx.markdown(
                    f'<div class="user-message">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                ctx.markdown(
                    f'<div class="bot-message">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
        
        ctx.markdown('</div>', unsafe_allow_html=True)
        
        # Create a form for the chat input
        with ctx.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Message RideWise Assistant...",
                key="user_message",
                label_visibility="collapsed"
            )
            
            send_button = st.form_submit_button("Send")
            
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
        
        ctx.markdown('</div>', unsafe_allow_html=True)
        
        # Add a clear chat button
        if ctx.button("Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

def display_chat_sidebar():
    """
    Display the chat interface in the sidebar.
    This is a convenience function that puts the chat in the sidebar.
    """
    display_chat_interface(st.sidebar)

def display_chat_expander():
    """
    Display the chat interface inside an expander.
    This keeps the chat hidden by default but easily accessible.
    """
    with st.expander("💬 RideWise Assistant", expanded=False):
        display_chat_interface()

def display_chat_tab():
    """
    Display the chat interface in a tab.
    This allows the chat to be part of a tabbed interface.
    """
    display_chat_interface()