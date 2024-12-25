# tests/test_llm_processor.py
import pytest
from ..llm.model import LlamaProcessor, ExtractionResult

def test_llm_processor(test_config):
    """Test LLM processor."""
    processor = LlamaProcessor(test_config)
    
    test_text = "John Smith works at Apple Inc. in California."
    doc = type('Document', (), {
        'chunks': [{'text': test_text, 'chunk_id': 'test_chunk'}]
    })
    
    results = processor.process_document(doc)
    
    assert len(results) > 0
    assert isinstance(results[0], ExtractionResult)
    assert len(results[0].entities) > 0

