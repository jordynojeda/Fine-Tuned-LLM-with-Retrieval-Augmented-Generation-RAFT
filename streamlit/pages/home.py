import streamlit as st
from pathlib import Path

def create_metrics_section():
    """Create the metrics display section."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", "Llama 3.1 8B", "Fine-tuned")

    with col2:
        st.metric("Chat Format", "ChatML", "Optimized")

    with col3:
        st.metric("Context Length", "10,048", "tokens")

def create_sidebar_content():
    """Create the sidebar content."""
    with st.sidebar:
        st.header("🏠 Home")
        st.write("Welcome to the Financial Advisor LLM application!")
        
        st.header("📋 Navigation")
        st.write("Use the pages above to:")
        st.write("• **Fine-tuned LLM** - Direct model interaction")
        st.write("• **Fine-tuned LLM + RAG** - Enhanced with knowledge retrieval")
        
        st.header("ℹ️ About")
        st.write("This application demonstrates the power of fine-tuned language models for financial advisory services.")
        
        st.header("🔧 Technical Details")
        st.write("- **Model**: Meta-Llama-3.1-8B-Instruct")
        st.write("- **Quantization**: Q4_K_M GGUF")
        st.write("- **Embedding**: all-MiniLM-L6-v2")
        st.write("- **Vector DB**: FAISS")
        st.write("- **Framework**: Streamlit")

def create_welcome_content():
    """Create the main welcome content."""
    st.markdown("""
    ## Welcome to your Financial Advisor AI Assistant

    This application provides two powerful ways to interact with our fine-tuned financial advisor model:

    ### Fine-tuned LLM
    - Direct interaction with our specialized financial advisor model
    - Fast responses based on the model's training
    - Great for general financial questions and advice

    ### Fine-tuned LLM + RAG
    - Enhanced with Retrieval-Augmented Generation
    - Accesses curated financial knowledge base
    - More accurate and detailed responses with source citations
    - Perfect for complex financial topics requiring specific documentation

    ---

    ### Getting Started
    1. Choose a page from the sidebar navigation
    2. Start asking financial questions
    3. Compare responses between the two approaches

    ### Features
    - **Real-time chat interface** - Natural conversation flow
    - **Source citations** - See where information comes from (RAG model) and view the source documents in the PDF viewer.
    - **Chat history** - Keep track of your conversations
    - **Knowledge base insights** - View retrieved context for transparency
    """)

def main():
    """Main function for the home page."""
    # Main page content
    st.title("💰 Financial Advisor LLM")
    st.markdown("---")

    # Welcome message
    create_welcome_content()
    
    # Add metrics section
    create_metrics_section()
    
    # Create sidebar content
    create_sidebar_content()

if __name__ == "__main__":
    main()