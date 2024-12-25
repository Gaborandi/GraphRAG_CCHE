# document_processor/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Document:
    """Base document class to store extracted text and metadata."""
    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
        source_path: Optional[Union[str, Path]] = None
    ):
        self.content = content
        self.metadata = metadata
        self.doc_id = doc_id or str(hash(content))
        self.source_path = source_path
        self.chunks: List[Dict[str, Any]] = []

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, source={self.source_path})"

class DocumentProcessor(ABC):
    """Abstract base class for document processors."""
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process(self, file_path: Union[str, Path]) -> Document:
        """Process a document and return a Document instance."""
        pass

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        chunks = []
        text_length = len(text)
        start = 0
        
        while start < text_length:
            end = min(start + self.config.chunk_size, text_length)
            
            # Adjust chunk end to avoid splitting words
            if end < text_length:
                while end > start and not text[end].isspace():
                    end -= 1
                if end == start:
                    end = min(start + self.config.chunk_size, text_length)
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start_idx': start,
                    'end_idx': end,
                    'chunk_id': f"chunk_{len(chunks)}",
                })
            
            start = end - self.config.overlap
        
        return chunks
