import streamlit as st
import sys
import os
from pathlib import Path

# Add the parent directory to the path to import shared utilities
sys.path.append(str(Path(__file__).parent.parent))

from utils.shared_utils import (
    load_llm,
    initialize_chat_history,
    clear_chat_history,
    display_chat_history,
    add_message_to_history,
    get_chat_history
)

# Constants
SESSION_KEY = "chat_history_finetuned"
SYSTEM_MESSAGE = "You are a helpful and knowledgeable financial advisor. Provide clear, easy-to-understand answers using plain English. Do not use markdown—only plain text."

@st.cache_resource
def get_llm():
    """Load and cache the LLM model."""
    return load_llm()

def create_page_header():
    """Create the page header and description."""
    st.title("🧠 Fine-tuned Financial Advisor LLM")
    st.markdown("---")
    
    st.markdown("""
    ### Direct Fine-tuned Model Interaction

    This page allows you to chat directly with our fine-tuned Llama 3.1 8B model that has been specifically trained for financial advisory services.

    **Features:**
    - ⚡ Fast response times
    - 🎯 Specialized financial knowledge
    - 💬 Natural conversation flow
    - 📝 Persistent chat history

    **Best for:**
    - General financial questions
    - Quick advice and explanations
    - Basic financial calculations
    - Investment fundamentals
    """)

def create_sidebar():
    """Create the sidebar with model information and controls."""
    with st.sidebar:
        st.header("🧠 Fine-tuned LLM")
        st.write("Direct interaction with the fine-tuned model.")
        
        st.header("📊 Model Information")
        st.info("""
        **Model**: Meta-Llama-3.1-8B-Instruct
        **Quantization**: Q4_K_M GGUF
        **Context Length**: 10,048 tokens
        **Max Response**: 1048 tokens
        **Temperature**: 0.7
        """)
        
        st.header("🎯 Use Cases")
        st.write("""
        - General financial advice
        - Investment basics
        - Market explanations
        - Financial planning concepts
        - Quick calculations
        """)
        
        st.header("🔧 Controls")
        
        # Chat history management
        if st.button("🗑️ Clear Chat History", key="clear_finetuned"):
            clear_chat_history(SESSION_KEY, SYSTEM_MESSAGE)
            st.rerun()
        
        create_chat_statistics()
        
        st.header("💡 Tips")
        st.write("""
        - Be specific with your questions
        - Ask follow-up questions for clarity
        - Use the model for explanations and advice
        - Try different phrasings for better results
        """)

def create_chat_statistics():
    """Display chat statistics in the sidebar."""
    chat_history = get_chat_history(SESSION_KEY)
    if len(chat_history) > 1:  # Exclude system message
        user_messages = len([msg for msg in chat_history if msg['role'] == 'user'])
        assistant_messages = len([msg for msg in chat_history if msg['role'] == 'assistant'])
        
        st.header("📈 Chat Statistics")
        st.write(f"**User messages**: {user_messages}")
        st.write(f"**Assistant responses**: {assistant_messages}")
        st.write(f"**Total exchanges**: {min(user_messages, assistant_messages)}")

def handle_chat_interaction(llm):
    """Handle the chat interface and user interactions."""
    st.header("💬 Chat Interface")
    
    # Display chat history
    display_chat_history(SESSION_KEY)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about finance...", key="finetuned_input")
    
    if user_input:
        # Add user message to history and display
        add_message_to_history(SESSION_KEY, "user", user_input)
        with st.chat_message("user"):
            st.text(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = llm.create_chat_completion(
                        messages=get_chat_history(SESSION_KEY),
                        max_tokens=1024,  # Quadruple the current limit
                        temperature=0.7,
                        stop=["<|eot_id|>"],  # Add more natural stops
                    )
                    assistant_msg = response['choices'][0]['message']['content']
                    st.text(assistant_msg)
                    
                    # Add assistant message to history
                    add_message_to_history(SESSION_KEY, "assistant", assistant_msg)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.write("I apologize, but I encountered an error. Please try again.")

def create_footer():
    """Create the page footer."""
    st.markdown("---")
    st.markdown("*This model has been fine-tuned specifically for financial advisory tasks and provides responses based on its training data.*")

def main():
    """Main function for the Fine-tuned LLM page."""
    # Page configuration
    st.set_page_config(
        page_title="Fine-tuned LLM",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize chat history
    initialize_chat_history(SESSION_KEY, SYSTEM_MESSAGE)
    
    # Load the model
    llm = get_llm()
    
    # Create page components
    create_page_header()
    handle_chat_interaction(llm)
    create_sidebar()
    create_footer()

if __name__ == "__main__":
    main()