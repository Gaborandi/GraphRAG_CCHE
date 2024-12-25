# utils/text_processing.py
import re
from typing import List, Dict, Any

class TextPreprocessor:
    """Text preprocessing utilities."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_metadata(text: str) -> Dict[str, Any]:
        """Extract basic metadata from text."""
        metadata = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
        }
        return metadata