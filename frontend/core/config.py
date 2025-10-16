import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_option = st.selectbox(
            "Select summarization model:",
            ["facebook/bart-large-cnn", "t5-small", "google/pegasus-xsum"],
            index=0
        )
        max_length = st.slider(
            "Max summary length:",
            min_value=50,
            max_value=300,
            value=150,
            step=10
        )
        min_length = st.slider(
            "Min summary length:",
            min_value=10,
            max_value=100,
            value=30,
            step=5
        )
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

    return {
        "model_option": model_option,
        "max_length": max_length,
        "min_length": min_length,
        "do_sample": do_sample,
        "temperature": temperature,
        "num_beams": num_beams
    }