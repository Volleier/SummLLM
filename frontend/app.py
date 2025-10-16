# app.py
import streamlit as st
import pandas as pd
import time

# Page configuration
st.set_page_config(
    page_title="Intelligent Text Summarization System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize model service
# @st.cache_resource
# def load_model_service():
#     """Cache the model service to avoid reloading"""
#     return SimpleModelService()

def main():
    # Title and description
    st.title("📝 LLM Intelligent Text Summarization")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_option = st.selectbox(
            "Select summarization model:",
            ["facebook/bart-large-cnn", "t5-small", "google/pegasus-xsum"],
            index=0
        )
        
        # Summary length controls
        max_length = st.slider(
            "Max summary length:",
            min_value=50,
            max_value=300,
            value=150,
            step=10,
            help="Control the maximum length of the generated summary"
        )
        
        min_length = st.slider(
            "Min summary length:",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Control the minimum length of the generated summary"
        )
        
        # Advanced options
        with st.expander("Advanced options"):
            do_sample = st.checkbox("Enable sampling", value=False)
            temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
            num_beams = st.slider("Number of beams", 1, 8, 4)
        
        st.markdown("---")
        st.info("""
        **Instructions:**
        1. Enter or paste text below
        2. Adjust parameters on the left
        3. Click the 'Generate Summary' button
        4. View results and evaluation metrics
        """)

    # Main content - two-column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input Text")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Direct input", "Example text", "Upload file"],
            horizontal=True
        )
        
        input_text = ""
        
        if input_method == "Direct input":
            input_text = st.text_area(
                "Enter text to summarize:",
                height=300,
                placeholder="Paste or type your text here...",
                help="Supports long text; the system will handle it automatically"
            )
            
        elif input_method == "Example text":
            example_options = {
                "News Article": """
                    At a recent international AI conference, researchers announced a major breakthrough.
                    They developed a new natural language processing model that demonstrates unprecedented
                    capabilities in text understanding and generation. This technology is expected to have
                    far-reaching impacts in machine translation, smart customer service, and content creation.
                    The team leader said the breakthrough came from improvements to the Transformer architecture,
                    allowing the model to better capture contextual semantics. The model has already achieved
                    state-of-the-art results on several benchmarks and is expected to be released to the developer
                    community next year.
                """,
                "Tech Report": """
                    With the rapid development of AI technologies, major tech companies are actively investing in AI.
                    Recently, a well-known tech company released its latest large language model with over a hundred
                    billion parameters, showing strong capabilities in natural language understanding, code generation,
                    and logical reasoning. Experts believe this marks a new phase in AI development and will have
                    significant effects across industries.
                """,
                "Academic Abstract": """
                    This paper investigates the application of deep learning-based summarization methods on large-scale datasets.
                    By comparing extractive and abstractive summarization strategies, we find that hybrid approaches combining
                    the strengths of both can achieve better results. Experimental results show that our method improves ROUGE
                    scores by 15% over baseline models. Additionally, we propose a new evaluation metric that more comprehensively
                    measures summary quality.
                """
            }
            
            selected_example = st.selectbox("Select example type:", list(example_options.keys()))
            input_text = st.text_area(
                "Example text:",
                value=example_options[selected_example],
                height=300
            )
            
        else:  # Upload file
            uploaded_file = st.file_uploader(
                "Upload text file",
                type=['txt', 'md'],
                help="Supports .txt and .md files"
            )
            if uploaded_file is not None:
                input_text = uploaded_file.getvalue().decode("utf-8")
                st.success(f"File uploaded successfully! File size: {len(input_text)} characters")

    with col2:
        st.subheader("📤 Summary Results")
        
        # Generate summary button
        if st.button(
            "🚀 Generate Summary", 
            type="primary", 
            use_container_width=True,
            disabled=not input_text.strip()
        ):
            if input_text.strip():
                with st.spinner("🤖 AI is analyzing the text and generating a summary..."):
                    try:
                        # Load model service
                        # service = load_model_service()
                        
                        # Show progress bar
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Generate summary
                        start_time = time.time()
                        summary = "This is a temporary summary."
                        # summary = service.summarize(
                           # input_text,
                           # max_length=max_length,
                           # min_length=min_length
                        # )
                        
                        processing_time = time.time() - start_time
                        
                        progress_bar.empty()
                        
                        # Display summary result
                        st.success("✅ Summary generation complete!")
                        
                        # Summary output
                        st.text_area(
                            "Generated summary:",
                            value=summary,
                            height=200,
                            key="summary_output"
                        )
                        
                        # Statistics
                        st.subheader("📊 Statistics")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("Original length", f"{len(input_text)} characters")
                        with col_stat2:
                            st.metric("Summary length", f"{len(summary)} characters")
                        with col_stat3:
                            compression_ratio = len(input_text) / len(summary) if len(summary) > 0 else 0
                            st.metric("Compression ratio", f"{compression_ratio:.1f}:1")
                        
                        # Processing time
                        st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download summary",
                            data=summary,
                            file_name="generated_summary.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
            else:
                st.warning("⚠️ Please enter text")

        # Placeholder when no summary generated
        else:
            st.info("👆 Please enter text on the left and click 'Generate Summary'")

    # Footer
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