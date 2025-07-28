import streamlit as st
import sys
import os
from pathlib import Path
import base64
import fitz  # PyMuPDF
from io import BytesIO
import re

# Add the parent directory to the path to import shared utilities
sys.path.append(str(Path(__file__).parent.parent))

from utils.shared_utils import (
    load_llm,
    load_rag_components,
    retrieve_relevant_docs,
    create_rag_prompt,
    initialize_chat_history,
    clear_chat_history,
    display_chat_history,
    add_message_to_history,
    get_chat_history
)

# Constants
SESSION_KEY = "chat_history_rag"
SYSTEM_MESSAGE = """You are a helpful and knowledgeable financial advisor with access to a curated knowledge base. Provide clear, easy-to-understand answers using plain English. Do not use markdown—only plain text. Always include citations to the source documents used in your responses."""

# PDF viewer functions
@st.cache_data
def get_pdf_page_image(pdf_path, page_number):
    """Extract a specific page from PDF as an image"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)  # PyMuPDF uses 0-based indexing
        
        # Render page to image
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        doc.close()
        return img_data
    except Exception as e:
        st.error(f"Error loading PDF page: {str(e)}")
        return None

def display_pdf_page(pdf_path, page_number, source_file):
    """Display a specific PDF page with highlighting"""
    img_data = get_pdf_page_image(pdf_path, page_number)
    
    if img_data:
        st.markdown(f"### 📄 Source: {source_file} - Page {page_number}")
        
        # Convert to base64 for display
        img_base64 = base64.b64encode(img_data).decode()
        
        # Create HTML with scrollable container
        html = f"""
        <div style="border: 2px solid #ddd; border-radius: 8px; padding: 10px; margin: 10px 0; background: white;">
            <div style="max-height: 600px; overflow-y: auto; text-align: center;">
                <img src="data:image/png;base64,{img_base64}" 
                     style="max-width: 100%; height: auto; border-radius: 4px;">
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.error(f"Could not load page {page_number} from {source_file}")

def get_pdf_path(source_file):
    """Get the full path to the PDF file"""
    # Adjust this path based on your PDF storage location
    pdf_folder = Path("financial_advisor_documents")  # Modify this path as needed
    return pdf_folder / source_file

@st.cache_resource
def get_llm():
    """Load and cache the LLM model."""
    return load_llm()

@st.cache_resource
def get_rag_components():
    """Load and cache the RAG components."""
    return load_rag_components()

def create_page_header():
    """Create the page header and description."""
    st.title("🔍 Fine-tuned LLM with RAG & PDF Viewer")
    st.markdown("---")
    
    st.markdown("""
    ### Enhanced Model with Retrieval-Augmented Generation & Source Visualization

    This page combines our fine-tuned financial advisor model with a powerful RAG system and adds **PDF page viewing** 
    to show you exactly where the information comes from.

    **Features:**
    - 📚 Access to curated financial knowledge base
    - 🎯 Context-aware responses with source citations
    - 🔍 Transparent document retrieval process
    - 📄 **PDF page viewer** - see the exact source pages
    - 🧠 Enhanced accuracy through document grounding
    - 🖼️ Visual source verification

    **Best for:**
    - Complex financial topics requiring source verification
    - Research-backed answers with visual proof
    - Regulatory and compliance questions
    - Detailed financial analysis with citations
    """)

def display_retrieved_context(relevant_docs, relevant_metadata):
    """Display retrieved context with PDF viewer."""
    with st.expander("📚 Retrieved Context & Source Pages", expanded=False):
        if relevant_docs:
            st.write("**Source documents used for this response:**")
            
            # Create tabs for each source
            tab_names = [f"Source {i+1}" for i in range(len(relevant_docs))]
            tabs = st.tabs(tab_names)
            
            for i, (doc, metadata, tab) in enumerate(zip(relevant_docs, relevant_metadata, tabs)):
                with tab:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write(f"**📄 {metadata['source_file']}**")
                        st.write(f"**Page:** {metadata['page_number']}")
                        st.write(f"**Similarity:** {metadata['similarity_score']:.3f}")
                        st.write("**Extracted Text:**")
                        st.text_area("", doc, height=150, key=f"text_{i}")
                    
                    with col2:
                        st.write("**Source PDF Page:**")
                        pdf_path = get_pdf_path(metadata['source_file'])
                        if pdf_path.exists():
                            display_pdf_page(pdf_path, metadata['page_number'], metadata['source_file'])
                        else:
                            st.warning(f"PDF file not found: {pdf_path}")
                            st.info("Please ensure PDF files are in the correct folder")
        else:
            st.write("No relevant documents found in the knowledge base.")
            

def handle_chat_interaction(llm, faiss_index, chunk_metadata, knowledge_base):
    """Handle the chat interface and user interactions."""
    # Apply styling once
    #apply_text_consistency()
    
    st.header("💬 Enhanced Chat Interface")
    
    # Display chat history
    display_chat_history(SESSION_KEY)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about finance (with knowledge base support)...", key="rag_input")
    
    if user_input:
        # Add user message to history and display
        add_message_to_history(SESSION_KEY, "user", user_input)
        with st.chat_message("user"):
            st.text(user_input)
        
        # Generate response with RAG
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant information..."):
                try:
                    # Retrieve relevant documents
                    relevant_docs, relevant_metadata = retrieve_relevant_docs(
                        user_input, faiss_index, knowledge_base, chunk_metadata
                    )
                    
                    # Show retrieved context with PDF viewer
                    display_retrieved_context(relevant_docs, relevant_metadata)
                    
                    # Create RAG-enhanced prompt
                    if relevant_docs:
                        rag_prompt = create_rag_prompt(user_input, relevant_docs, relevant_metadata)
                        
                        # Create temporary message history for RAG
                        chat_history = get_chat_history(SESSION_KEY)
                        rag_messages = chat_history[:-1] + [
                            {"role": "user", "content": rag_prompt}
                        ]
                    else:
                        # No relevant documents found, use original query
                        rag_messages = get_chat_history(SESSION_KEY)
                    
                    # Generate response
                    response = llm.create_chat_completion(
                        messages=rag_messages,
                        max_tokens=1048,
                        temperature=0.7,
                        stop=["<|eot_id|>"],
                    )
                    assistant_msg = response['choices'][0]['message']['content']
                    st.text(assistant_msg)
                    
                    # Add assistant message to history
                    add_message_to_history(SESSION_KEY, "assistant", assistant_msg)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.write("I apologize, but I encountered an error. Please try again.")

def create_knowledge_base_status(knowledge_base, chunk_metadata):
    """Create the knowledge base status section."""
    st.header("📚 Knowledge Base Status")
    if knowledge_base:
        # Calculate statistics
        unique_sources = len(set(meta['source_file'] for meta in chunk_metadata))
        total_pages = len(set(f"{meta['source_file']}_{meta['page_number']}" for meta in chunk_metadata))
        
        st.success(f"""
        ✅ **Knowledge base loaded**
        - {len(knowledge_base)} text chunks
        - {unique_sources} PDF documents
        - {total_pages} total pages
        """)
        
    else:
        st.error("❌ Knowledge base not loaded")
        st.write("Please check the PDF documents folder.")

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

def create_sidebar(knowledge_base, chunk_metadata):
    """Create the sidebar with all components."""
    with st.sidebar:
        st.header("🔍 LLM + RAG + PDF Viewer")
        st.write("Enhanced model with knowledge base retrieval and PDF visualization.")
        
        st.header("📊 System Information")
        st.info(f"""
        **Model**: Meta-Llama-3.1-8B-Instruct
        **Embedding**: all-MiniLM-L6-v2
        **Vector DB**: FAISS
        **Documents**: {len(knowledge_base) if knowledge_base else 0}
        **Retrieval**: Top-3 similarity
        **PDF Viewer**: PyMuPDF
        """)
        
        create_knowledge_base_status(knowledge_base, chunk_metadata)
        
        st.header("🎯 Enhanced RAG Features")
        st.write("""
        - **Document Retrieval**: Finds relevant context
        - **Source Attribution**: Shows document sources
        - **Page References**: Links to specific pages
        - **PDF Viewer**: Visual source verification
        - **Similarity Scoring**: Relevance ranking
        - **Context Windowing**: Optimal chunk sizes
        """)
        
        st.header("🔧 Controls")
        
        # Chat history management
        if st.button("🗑️ Clear Chat History", key="clear_rag"):
            clear_chat_history(SESSION_KEY, SYSTEM_MESSAGE)
            st.rerun()
        
        create_chat_statistics()
        
        st.header("💡 Tips")
        st.write("""
        - Ask specific questions about financial topics
        - Check the Retrieved Context to see sources AND pages
        - Use the PDF viewer to verify information
        - Try queries about regulations, procedures, or specific financial concepts
        - The PDF viewer shows exactly where information comes from
        """)

def create_footer():
    """Create the page footer."""
    st.markdown("---")
    st.markdown("*This enhanced model combines fine-tuned financial knowledge with real-time document retrieval and PDF visualization for maximum transparency and accuracy.*")

def main():
    """Main function for the LLM + RAG page."""
    # Page configuration
    st.set_page_config(
        page_title="LLM + RAG",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize chat history
    initialize_chat_history(SESSION_KEY, SYSTEM_MESSAGE)
    
    # Load models and components
    llm = get_llm()
    faiss_index, chunk_metadata, knowledge_base = get_rag_components()
    
    # Create page components
    create_page_header()
    handle_chat_interaction(llm, faiss_index, chunk_metadata, knowledge_base)
    create_sidebar(knowledge_base, chunk_metadata)
    create_footer()

if __name__ == "__main__":
    main()