# utils/shared_utils.py
import streamlit as st
from llama_cpp import Llama
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from typing import List, Dict, Any, Tuple
import PyPDF2
import re
from pathlib import Path
from huggingface_hub import hf_hub_download

# Cache model to avoid reloading
@st.cache_resource
def load_llm():
    """Load the fine-tuned LLM model from Hugging Face Hub"""
    model_path = hf_hub_download(
        repo_id="jordynojeda/Meta-Llama-3.1-8B-Instruct-bnb-4bit-financial-advisor-qlora-GGUF",  # Change this
        filename="unsloth.Q4_K_M.gguf",
        local_dir="models",  # Optional local cache folder
        cache_dir="models"   # Optional: ensures reuse
    )
    
    return Llama(
        model_path=model_path,
        n_ctx=10048,
        n_gpu_layers=32,
        chat_format="chatml",
    )

# Cache embedding model
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model for embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF files with page information
def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from a PDF file, keeping track of page numbers"""
    pages = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    pages.append({
                        'page_number': page_num + 1,
                        'text': page_text.strip(),
                        'char_count': len(page_text.strip())
                    })
    except Exception as e:
        st.error(f"Error reading {pdf_path}: {str(e)}")
    return pages

# Function to chunk text within page boundaries
def chunk_text(pages: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk text while respecting page boundaries"""
    chunks = []
    
    for page_info in pages:
        page_num = page_info['page_number']
        page_text = page_info['text']
        
        # Clean up the text
        page_text = re.sub(r'\s+', ' ', page_text.strip())
        
        # If the entire page fits in one chunk, use it as is
        if len(page_text) <= chunk_size:
            chunks.append({
                'text': page_text,
                'page_number': page_num,
                'chunk_size': len(page_text),
                'is_full_page': True,
                'chunk_index_on_page': 0,
                'total_chunks_on_page': 1
            })
        else:
            # Split page into multiple chunks
            page_chunks = []
            start = 0
            chunk_index = 0
            
            while start < len(page_text):
                # Calculate end position for this chunk
                end = start + chunk_size
                
                # If we're at the end of the page text, take what's left
                if end >= len(page_text):
                    chunk_text = page_text[start:].strip()
                    if chunk_text:
                        page_chunks.append({
                            'text': chunk_text,
                            'page_number': page_num,
                            'chunk_size': len(chunk_text),
                            'is_full_page': False,
                            'chunk_index_on_page': chunk_index,
                            'start_pos': start,
                            'end_pos': len(page_text)
                        })
                    break
                
                # Find the best break point within the chunk
                chunk_text = page_text[start:end]
                
                # Try to find sentence boundary (. ! ?) followed by space
                sentence_breaks = []
                for match in re.finditer(r'[.!?]\s+', chunk_text):
                    sentence_breaks.append(match.end())
                
                # If we found sentence breaks, use the last one that's not too close to the start
                if sentence_breaks:
                    best_break = sentence_breaks[-1]
                    if best_break > chunk_size * 0.3:  # At least 30% of chunk size
                        end = start + best_break
                    else:
                        # Try word boundary
                        word_match = re.search(r'\s+(?=\S)', chunk_text[::-1])
                        if word_match:
                            end = start + chunk_size - word_match.start()
                else:
                    # No sentence breaks found, try to break at word boundary
                    word_match = re.search(r'\s+(?=\S)', chunk_text[::-1])
                    if word_match:
                        end = start + chunk_size - word_match.start()
                
                # Extract the chunk
                chunk_text = page_text[start:end].strip()
                if chunk_text:
                    page_chunks.append({
                        'text': chunk_text,
                        'page_number': page_num,
                        'chunk_size': len(chunk_text),
                        'is_full_page': False,
                        'chunk_index_on_page': chunk_index,
                        'start_pos': start,
                        'end_pos': end
                    })
                    chunk_index += 1
                
                # Calculate next start position with overlap (but stay within page)
                next_start = max(start + 1, end - overlap)
                
                # If we're not making progress, force move forward
                if next_start <= start:
                    next_start = start + chunk_size // 2
                
                start = next_start
            
            # Add total chunks info to all chunks from this page
            for chunk in page_chunks:
                chunk['total_chunks_on_page'] = len(page_chunks)
            
            chunks.extend(page_chunks)
    
    return chunks

# Updated function to process PDF documents with page-aware chunking
def process_pdf_documents(pdf_folder: str, chunk_size: int = 1000, overlap: int = 200) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Process all PDF documents in the folder with page-aware chunking"""
    pdf_folder = Path(pdf_folder)
    
    if not pdf_folder.exists():
        st.error(f"PDF folder '{pdf_folder}' does not exist.")
        return [], []
    
    all_chunks = []
    chunk_metadata = []
    
    # Find all PDF files
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_files:
        st.warning(f"No PDF files found in '{pdf_folder}'")
        return [], []
    
    st.info(f"Processing {len(pdf_files)} PDF files with page-aware chunking...")
    
    # Track statistics
    total_chunks = 0
    total_pages = 0
    oversized_chunks = 0
    
    # Process each PDF
    for pdf_file in pdf_files:
        st.write(f"Processing: {pdf_file.name}")
        
        # Extract text with page information
        pages = extract_text_from_pdf(str(pdf_file))
        
        if not pages:
            st.warning(f"No text extracted from {pdf_file.name}")
            continue
        
        total_pages += len(pages)
        st.write(f"  → {len(pages)} pages found")
        
        # Create chunks respecting page boundaries
        chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)
        
        # Validate chunk sizes and create metadata
        file_oversized = 0
        for global_chunk_index, chunk in enumerate(chunks):
            if chunk['chunk_size'] > chunk_size * 1.5:  # Allow 50% tolerance
                file_oversized += 1
                oversized_chunks += 1
            
            # Create comprehensive metadata
            chunk_metadata.append({
                'source_file': pdf_file.name,
                'global_chunk_index': global_chunk_index,
                'page_number': chunk['page_number'],
                'chunk_index_on_page': chunk['chunk_index_on_page'],
                'total_chunks_on_page': chunk['total_chunks_on_page'],
                'is_full_page': chunk['is_full_page'],
                'chunk_size': chunk['chunk_size'],
                'is_oversized': chunk['chunk_size'] > chunk_size * 1.5,
                'start_pos': chunk.get('start_pos', 0),
                'end_pos': chunk.get('end_pos', chunk['chunk_size'])
            })
            
            all_chunks.append(chunk['text'])
        
        if file_oversized > 0:
            st.warning(f"⚠️ {file_oversized} chunks in {pdf_file.name} exceed size limit")
        
        total_chunks += len(chunks)
        st.write(f"  → {len(chunks)} chunks created from {len(pages)} pages")
    
    # Display summary statistics
    st.success(f"✅ Processing complete!")
    st.info(f"""
    **Summary Statistics:**
    - Total pages processed: {total_pages}
    - Total chunks created: {total_chunks}
    - Average chunks per page: {total_chunks/total_pages:.1f}
    - Average chunk size: {sum(len(chunk) for chunk in all_chunks) / len(all_chunks):.0f} characters
    - Oversized chunks: {oversized_chunks} ({oversized_chunks/total_chunks*100:.1f}%)
    """)
    
    # Show chunk size distribution
    chunk_sizes = [len(chunk) for chunk in all_chunks]
    if chunk_sizes:
        st.write(f"**Chunk size distribution:**")
        st.write(f"- Min: {min(chunk_sizes)} chars")
        st.write(f"- Max: {max(chunk_sizes)} chars")
        st.write(f"- Median: {np.median(chunk_sizes):.0f} chars")
    
    # Show page distribution
    page_counts = {}
    for metadata in chunk_metadata:
        file_name = metadata['source_file']
        if file_name not in page_counts:
            page_counts[file_name] = set()
        page_counts[file_name].add(metadata['page_number'])
    
    st.write("**Page distribution by file:**")
    for file_name, pages in page_counts.items():
        st.write(f"- {file_name}: {len(pages)} pages")
    
    return chunk_metadata, all_chunks

# Cache RAG components
@st.cache_resource
def load_rag_components():
    """Load or create RAG components (knowledge base, embeddings, FAISS index)"""
    
    # Check if RAG components exist and are up to date
    rag_data_path = "./rag_data"
    index_path = f"{rag_data_path}/faiss_index.index"
    metadata_path = f"{rag_data_path}/chunk_metadata.pkl"
    documents_path = f"{rag_data_path}/documents.pkl"
    
    # Check if PDF folder exists
    pdf_folder = "./financial_advisor_documents"
    if not os.path.exists(pdf_folder):
        st.error(f"PDF folder '{pdf_folder}' does not exist. Please create it and add PDF files.")
        return None, [], []
    
    # Check if we need to rebuild (if PDFs are newer than cached data)
    rebuild_needed = True
    if os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(documents_path):
        try:
            # Get the latest modification time of PDFs
            pdf_files = list(Path(pdf_folder).glob("*.pdf"))
            if pdf_files:
                latest_pdf_time = max(pdf_file.stat().st_mtime for pdf_file in pdf_files)
                index_time = os.path.getmtime(index_path)
                
                if index_time > latest_pdf_time:
                    rebuild_needed = False
        except:
            rebuild_needed = True
    
    if not rebuild_needed:
        # Load existing components
        try:
            index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                chunk_metadata = pickle.load(f)
            with open(documents_path, "rb") as f:
                documents = pickle.load(f)
            st.success("Loaded cached RAG components")
            return index, chunk_metadata, documents
        except:
            rebuild_needed = True
    
    if rebuild_needed:
        st.info("Building RAG components from PDF documents...")
        
        # Process PDF documents
        chunk_metadata, documents = process_pdf_documents(pdf_folder)
        
        if not documents:
            st.error("No documents were processed. Please check your PDF files.")
            return None, [], []
        
        # Create embeddings
        embedding_model = load_embedding_model()
        st.write("Creating embeddings...")
        embeddings = embedding_model.encode(documents, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save components
        os.makedirs(rag_data_path, exist_ok=True)
        faiss.write_index(index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(chunk_metadata, f)
        with open(documents_path, "wb") as f:
            pickle.dump(documents, f)
        
        st.success(f"Successfully processed {len(documents)} chunks from {len(set(meta['source_file'] for meta in chunk_metadata))} PDF files")
        
        return index, chunk_metadata, documents

def retrieve_relevant_docs(query: str, index, documents: List[str], chunk_metadata: List[Dict[str, Any]], k: int = 3) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Retrieve relevant documents based on query similarity"""
    if index is None or not documents:
        return [], []
    
    embedding_model = load_embedding_model()
    query_embedding = embedding_model.encode([query])
    
    # Search for similar documents
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    # Return relevant documents with metadata
    relevant_docs = []
    relevant_metadata = []
    
    for i, idx in enumerate(indices[0]):
        if idx < len(documents):  # Safety check
            relevant_docs.append(documents[idx])
            metadata = chunk_metadata[idx].copy()
            metadata['similarity_score'] = float(distances[0][i])
            relevant_metadata.append(metadata)
    
    return relevant_docs, relevant_metadata

def create_rag_prompt(query: str, relevant_docs: List[str], relevant_metadata: List[Dict[str, Any]]) -> str:
    """Create a prompt that includes relevant context with page and source information"""
    context_parts = []
    
    for doc, metadata in zip(relevant_docs, relevant_metadata):
        page_info = f"Page {metadata['page_number']}"
        if not metadata['is_full_page']:
            page_info += f" (Part {metadata['chunk_index_on_page']+1}/{metadata['total_chunks_on_page']})"
        
        source_info = f"[Source: {metadata['source_file']}, {page_info}]"
        context_parts.append(f"{source_info}\n{doc}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Based on the following financial knowledge from our documents:

{context}

Please answer the following question: {query}

Use the provided context to give a comprehensive and accurate answer. Reference the source documents and page numbers when relevant. If the context doesn't fully cover the question, you may use your general knowledge but prioritize the provided information."""
    
    return prompt

def initialize_chat_history(session_key: str, system_message: str):
    """Initialize chat history for a specific session"""
    if session_key not in st.session_state:
        st.session_state[session_key] = [
            {"role": "system", "content": system_message}
        ]

def clear_chat_history(session_key: str, system_message: str):
    """Clear chat history for a specific session"""
    st.session_state[session_key] = [
        {"role": "system", "content": system_message}
    ]

def display_chat_history(session_key: str):
    """Display chat history (skip system message)"""
    if session_key in st.session_state:
        for msg in st.session_state[session_key][1:]:
            st.chat_message(msg["role"]).text(msg["content"])

def add_message_to_history(session_key: str, role: str, content: str):
    """Add a message to chat history"""
    if session_key not in st.session_state:
        st.session_state[session_key] = []
    st.session_state[session_key].append({"role": role, "content": content})

def get_chat_history(session_key: str) -> List[Dict[str, str]]:
    """Get chat history for a specific session"""
    return st.session_state.get(session_key, [])