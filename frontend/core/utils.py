def load_uploaded_text(uploaded_file):
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except Exception:
        # 备用：直接返回 bytes 的 repr
        return str(uploaded_file.getvalue())

def compute_metrics(original, summary):
    """简单统计，若需要可扩展"""
    orig_len = len(original)
    summ_len = len(summary)
    ratio = orig_len / summ_len if summ_len > 0 else 0
    return {"original_len": orig_len, "summary_len": summ_len, "compression_ratio": ratio}