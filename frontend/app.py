import streamlit as st
from core.config import render_sidebar
from ui.input import render_input_column
from ui.result import render_result_column
from services.model_service import SimpleModelService

def main():
    """Streamlit application entry point."""
    # Page configuration: title, icon, layout and initial sidebar state
    st.set_page_config(
        page_title="Intelligent Text Summarization System",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main page title and divider
    st.title("LLM Intelligent Text Summarization")
    st.markdown("---")

    # Render sidebar and get configuration
    config = render_sidebar()
    # Initialize model service (used for generating summaries, etc.)
    service = SimpleModelService()

    # Create two-column layout: left for input, right for results
    col1, col2 = st.columns([1, 1])
    with col1:
        # Render input column and get input text
        input_text = render_input_column()
    with col2:
        # Render result column, passing input text, config, and model service
        render_result_column(input_text, config, service)

    # Footer note
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Transformer-based Intelligent Text Summarization System | Supports long-text processing | Real-time generation</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
