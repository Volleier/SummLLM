import re
from backend.config import settings

def preprocess_text(text: str, max_input_length: int | None = None) -> str:
    """Simple text cleaning and length truncation."""
    if max_input_length is None:
        max_input_length = settings.MAX_INPUT_LENGTH
    
    text = re.sub(r"\s+", " ", text)  # collapse whitespaces
    text = re.sub(r"[^\w\s.,!?;:()\-]", "", text)  # remove special chars
    text = text.strip()

    if len(text) > max_input_length:
        text = text[:max_input_length] + "..."
    return text