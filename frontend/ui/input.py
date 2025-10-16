import streamlit as st
from core.utils import load_uploaded_text

def render_input_column():
    st.subheader("📥 Input Text")
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
            placeholder="Paste or type your text here..."
        )
    elif input_method == "Example text":
        example_options = {
            "News Article": """
                At a recent international AI conference, researchers announced a major breakthrough.
                They developed a new natural language processing model that demonstrates unprecedented
                capabilities in text understanding and generation. This technology is expected to have
                far-reaching impacts in machine translation, smart customer service, and content creation.
            """,
            "Tech Report": """
                With the rapid development of AI technologies, major tech companies are actively investing in AI.
                Recently, a well-known tech company released its latest large language model with over a hundred
                billion parameters.
            """,
            "Academic Abstract": """
                This paper investigates the application of deep learning-based summarization methods on large-scale datasets.
            """
        }
        selected_example = st.selectbox("Select example type:", list(example_options.keys()))
        input_text = st.text_area("Example text:", value=example_options[selected_example], height=300)
    else:
        uploaded_file = st.file_uploader("Upload text file", type=['txt', 'md'])
        if uploaded_file is not None:
            input_text = load_uploaded_text(uploaded_file)
            st.success(f"File uploaded successfully! File size: {len(input_text)} characters")

    return input_text