import html
import re

def clean_text(text: str) -> str:
    """
    Normalizes text for downstream tokenization while preserving character positions
    as closely as possible.
    """
    if not isinstance(text, str):
        return ""
    
    # Decode HTML/XML entities like &quot;, &amp;
    text = html.unescape(text)
    
    # Replace non-breaking spaces and unusual whitespace with standard space
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()