# orchestration/pipeline.py
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time
from dataclasses import dataclass
import json

from ..config import Config
from ..document_processor.factory import DocumentProcessorFactory
from ..document_processor.base import Document
from ..llm.model import LlamaProcessor, ExtractionResult
from ..database.graph import KnowledgeGraph

@dataclass
class ProcessingResult:
    """Container for document processing results."""
    doc_id: str
    success: bool
    error: Optional[str] = None
    entities: Optional[List[Dict[str, Any]]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    processing_time: float = 0.0

class KnowledgeGraphPipeline:
    """Orchestrates the knowledge graph creation pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.llm_processor = LlamaProcessor(config)
        self.knowledge_graph = KnowledgeGraph(config)
        
        # Thread-safe queues for processing
        self.document_queue = Queue()
        self.extraction_queue = Queue()
        
        # Processing status tracking
        self.processing_status: Dict[str, ProcessingResult] = {}
        self._status_lock = threading.Lock()

    def process_documents(self, file_paths: List[Union[str, Path]], 
                        max_workers: int = 4) -> Dict[str, ProcessingResult]:
        """Process multiple documents in parallel."""
        start_time = time.time()
        self.logger.info(f"Starting batch processing of {len(file_paths)} documents")
        
        try:
            # Initialize processing status
            for file_path in file_paths:
                doc_id = str(Path(file_path).stem)
                self.processing_status[doc_id] = ProcessingResult(
                    doc_id=doc_id,
                    success=False
                )
            
            # Process documents in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit document processing tasks
                future_to_path = {
                    executor.submit(self._process_single_document, path): path
                    for path in file_paths
                }
                
                # Process results as they complete
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        self._update_processing_status(result)
                    except Exception as e:
                        self.logger.error(f"Error processing {path}: {str(e)}")
                        doc_id = str(Path(path).stem)
                        self._update_processing_status(ProcessingResult(
                            doc_id=doc_id,
                            success=False,
                            error=str(e)
                        ))
            
            # Calculate total processing time
            total_time = time.time() - start_time
            self.logger.info(f"Batch processing completed in {total_time:.2f} seconds")
            
            return self.processing_status
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            raise

    def _process_single_document(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Process a single document through the pipeline."""
        start_time = time.time()
        file_path = Path(file_path)
        doc_id = str(file_path.stem)
        
        try:
            # 1. Document Processing
            document = self._process_document(file_path)
            
            # 2. LLM Processing
            extraction_results = self._process_text(document)
            
            # 3. Knowledge Graph Integration
            self._integrate_to_graph(doc_id, extraction_results)
            
            # 4. Prepare result summary
            entities = []
            relationships = []
            
            for result in extraction_results:
                entities.extend(result.entities)
                relationships.extend(result.relationships)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                doc_id=doc_id,
                success=True,
                entities=entities,
                relationships=relationships,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            return ProcessingResult(
                doc_id=doc_id,
                success=False,
                error=str(e)
            )

    def _process_document(self, file_path: Path) -> Document:
        """Process document using appropriate processor."""
        self.logger.info(f"Processing document: {file_path}")
        processor = DocumentProcessorFactory.get_processor(file_path, self.config)
        return processor.process(file_path)

    def _process_text(self, document: Document) -> List[ExtractionResult]:
        """Process document text using LLM."""
        self.logger.info(f"Processing text for document: {document.doc_id}")
        return self.llm_processor.process_document(document)

    def _integrate_to_graph(self, doc_id: str, results: List[ExtractionResult]):
        """Integrate extraction results into knowledge graph."""
        self.logger.info(f"Integrating results for document: {doc_id}")
        self.knowledge_graph.process_extraction_results(doc_id, results)

    def _update_processing_status(self, result: ProcessingResult):
        """Update processing status in thread-safe manner."""
        with self._status_lock:
            self.processing_status[result.doc_id] = result

class PipelineManager:
    """Manages pipeline execution and provides high-level interface."""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config = Config(config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pipeline = KnowledgeGraphPipeline(self.config)

    def process_directory(self, directory_path: Union[str, Path], 
                        file_pattern: str = "*.*") -> Dict[str, Any]:
        """Process all matching files in a directory."""
        directory_path = Path(directory_path)
        file_paths = []
        
        # Collect all matching files
        for pattern in file_pattern.split(","):
            file_paths.extend(directory_path.glob(pattern.strip()))
        
        if not file_paths:
            raise ValueError(f"No matching files found in {directory_path}")
        
        # Process files
        results = self.pipeline.process_documents(file_paths)
        
        # Prepare summary
        summary = self._prepare_processing_summary(results)
        
        return summary

    def _prepare_processing_summary(self, results: Dict[str, ProcessingResult]) -> Dict[str, Any]:
        """Prepare summary of processing results."""
        summary = {
            'total_documents': len(results),
            'successful': sum(1 for r in results.values() if r.success),
            'failed': sum(1 for r in results.values() if not r.success),
            'total_entities': sum(len(r.entities or []) for r in results.values()),
            'total_relationships': sum(len(r.relationships or []) for r in results.values()),
            'total_processing_time': sum(r.processing_time for r in results.values()),
            'documents': {
                doc_id: {
                    'success': result.success,
                    'error': result.error,
                    'processing_time': result.processing_time,
                    'entity_count': len(result.entities or []),
                    'relationship_count': len(result.relationships or [])
                }
                for doc_id, result in results.items()
            }
        }
        
        return summary
