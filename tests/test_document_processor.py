# tests/test_document_processor.py
import pytest
from ..document_processor.processors import PDFProcessor, DocxProcessor, CSVProcessor
from ..document_processor.factory import DocumentProcessorFactory

def test_pdf_processor(test_config, test_documents):
    """Test PDF document processor."""
    processor = PDFProcessor(test_config)
    document = processor.process(test_documents['pdf'])
    
    assert document is not None
    assert document.content
    assert document.metadata['file_type'] == 'pdf'
    assert len(document.chunks) > 0

def test_docx_processor(test_config, test_documents):
    """Test DOCX document processor."""
    processor = DocxProcessor(test_config)
    document = processor.process(test_documents['docx'])
    
    assert document is not None
    assert document.content
    assert document.metadata['file_type'] == 'docx'
    assert len(document.chunks) > 0

