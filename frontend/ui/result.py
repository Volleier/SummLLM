import streamlit as st
import time
from core.utils import compute_metrics

def render_result_column(input_text, config, service):
    st.subheader("📤 Summary Results")

    can_generate = bool(input_text and input_text.strip())
    if st.button("🚀 Generate Summary", type="primary", use_container_width=True, disabled=not can_generate):
        if not can_generate:
            st.warning("⚠️ Please enter text")
            return

        with st.spinner("🤖 AI is analyzing the text and generating a summary..."):
            try:
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

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
                progress_bar.empty()

                st.success("✅ Summary generation complete!")
                st.text_area("Generated summary:", value=summary, height=200, key="summary_output")

                st.subheader("📊 Statistics")
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Original length", f"{len(input_text)} characters")
                with col_stat2:
                    st.metric("Summary length", f"{len(summary)} characters")
                with col_stat3:
                    compression_ratio = len(input_text) / len(summary) if len(summary) > 0 else 0
                    st.metric("Compression ratio", f"{compression_ratio:.1f}:1")

                st.info(f"⏱️ Processing time: {processing_time:.2f} seconds")
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
        st.info("👆 Please enter text on the left and click 'Generate Summary'")