import streamlit as st

def main():
    """Main function to set up navigation and run the app."""
    # Initialize page config
    st.set_page_config(
        page_title="Financial Advisor LLM",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Define pages with custom titles
    home = st.Page("pages/home.py", title="Home")
    fine_tuned_llm = st.Page("pages/fine_tuned_llm.py", title="Fine-tuned LLM") 
    fine_tuned_llm_rag = st.Page("pages/fine_tuned_and_rag.py", title="Fine-tuned LLM + RAG (RAFT)")

    # Create navigation
    pg = st.navigation([home, fine_tuned_llm, fine_tuned_llm_rag])
    pg.run()

if __name__ == "__main__":
    main()