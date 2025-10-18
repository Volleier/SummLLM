import streamlit as st
import time
from core.utils import compute_metrics  # keep import (enable if you need statistical functions)

def render_result_column(input_text, config, service):
    # Render the right-side results column: summary generation, statistics, and download
    st.subheader("Summary Results")

    # Determine whether there is text available to generate a summary
    can_generate = bool(input_text and input_text.strip())
    # Generate button (disabled when no input or characters <= 100)
    if 'generate_disabled' not in st.session_state:
        st.session_state['generate_disabled'] = True

    generate_disabled = st.session_state.get('generate_disabled', True)
    generate_button = st.button("Generate Summary", disabled=generate_disabled, type="primary", use_container_width=True)

    if generate_button:
        if not can_generate:
            st.warning("Please enter text")
            return

        # Show waiting indicator
        with st.spinner("AI is analyzing the text and generating a summary..."):
            try:
                # Simple progress bar animation (for demonstration only)
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                # Call the service to generate the summary and measure time
                start_time = time.time()
                summary = service.summarize(
                    input_text,
                    max_length=config["max_length"],
                    min_length=config["min_length"],
                    do_sample=config["do_sample"],
                    temperature=config["temperature"],
                    num_beams=config["num_beams"]
                )
                processing_time = time.time() - start_time
                progress_bar.empty()  # clear the progress bar

                # Display result area: success message and summary text box
                st.success("Summary generation complete!")
                st.text_area("Generated summary:", value=summary, height=200, key="summary_output")

                # Display statistics (original length, summary length, compression ratio)
                st.subheader("Statistics")
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Original length", f"{len(input_text)} characters")
                with col_stat2:
                    st.metric("Summary length", f"{len(summary)} characters")
                with col_stat3:
                    compression_ratio = len(input_text) / len(summary) if len(summary) > 0 else 0
                    st.metric("Compression ratio", f"{compression_ratio:.1f}:1")

                # Show processing time and provide a download button
                st.info(f"Processing time: {processing_time:.2f} seconds")
                st.download_button(
                    label="Download summary",
                    data=summary,
                    file_name="generated_summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                # Catch and display errors during generation
                st.error(f"Error generating summary: {str(e)}")
    else:
        # Initial hint: instruct the user to enter text on the left and click generate
        st.info("Please enter text on the left and click 'Generate Summary'")