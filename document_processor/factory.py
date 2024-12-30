# document_processor/factory.py
from pathlib import Path
from typing import Union

from config import Config
from document_processor.base import DocumentProcessor, Document

class DocumentProcessorFactory:
    """Factory for creating document processors based on file type."""
    
    @staticmethod
    def get_processor(file_path: Union[str, Path], config: Config) -> DocumentProcessor:
        """Get appropriate processor for the file type."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return PDFProcessor(config)
        elif suffix == '.docx':
            return DocxProcessor(config)
        elif suffix == '.csv':
            return CSVProcessor(config)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def process_document(file_path: Union[str, Path], config: Config) -> Document:
        """Process a document using the appropriate processor."""
        processor = DocumentProcessorFactory.get_processor(file_path, config)
        return processor.process(file_path)