import streamlit as st
from core.config import render_sidebar
from ui.input import render_input_column
from ui.result import render_result_column
from services.model_service import SimpleModelService

def main():
    st.set_page_config(
        page_title="Intelligent Text Summarization System",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📝 LLM Intelligent Text Summarization")
    st.markdown("---")

    # 渲染侧边并获取配置
    config = render_sidebar()

    # 初始化模型服务（可替换为实际实现）
    service = SimpleModelService()

    # 两列布局
    col1, col2 = st.columns([1, 1])
    with col1:
        input_text = render_input_column()
    with col2:
        render_result_column(input_text, config, service)

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