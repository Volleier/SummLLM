import re

def preprocess_text(text: str, max_input_length: int = 1024) -> str:
    """Simple text cleaning and length truncation."""
    text = re.sub(r"\s+", " ", text)  # collapse whitespaces
    text = re.sub(r"[^\w\s.,!?;:()\ -]", "", text)  # remove special chars
    text = text.strip()

    if len(text) > max_input_length:
        text = text[:max_input_length] + "..."
    return text