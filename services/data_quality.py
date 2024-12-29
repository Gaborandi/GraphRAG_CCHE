# services/embedding_manager.py
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
import sqlite3
import json
import threading
from queue import Queue
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

from ..config import Config

@dataclass
class EmbeddingJob:
    """Container for embedding generation job."""
    id: str
    text: str
    metadata: Dict[str, Any]
    timestamp: datetime
    priority: int = 0

@dataclass
class EmbeddingResult:
    """Container for embedding generation result."""
    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    generation_time: float
    error: Optional[str] = None

class EmbeddingCache:
    """SQLite-based embedding cache."""
    
    def __init__(self, cache_path: Union[str, Path]):
        self.cache_path = Path(cache_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize cache
        self._init_cache()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'size': 0
        }
        self._stats_lock = threading.Lock()

    def _init_cache(self):
        """Initialize cache database."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.cursor()
                
                # Create tables if they don't exist
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    embedding BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
                """)
                
                # Create indexes
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON embeddings(last_accessed)
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing cache: {str(e)}")
            raise

    def get(self, text: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get embedding from cache."""
        embedding_id = self._generate_id(text)
        
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.cursor()
                
                # Get embedding
                cursor.execute("""
                SELECT embedding, metadata, access_count 
                FROM embeddings 
                WHERE id = ?
                """, (embedding_id,))
                
                result = cursor.fetchone()
                
                if result:
                    # Update access statistics
                    cursor.execute("""
                    UPDATE embeddings 
                    SET last_accessed = CURRENT_TIMESTAMP,
                        access_count = access_count + 1
                    WHERE id = ?
                    """, (embedding_id,))
                    
                    conn.commit()
                    
                    # Update cache statistics
                    with self._stats_lock:
                        self.stats['hits'] += 1
                    
                    # Convert blob to numpy array
                    embedding_bytes = result[0]
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    metadata = json.loads(result[1])
                    
                    return embedding, metadata
                else:
                    with self._stats_lock:
                        self.stats['misses'] += 1
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def put(self, text: str, embedding: np.ndarray,
            metadata: Dict[str, Any]):
        """Store embedding in cache."""
        embedding_id = self._generate_id(text)
        
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.cursor()
                
                # Convert numpy array to blob
                embedding_bytes = embedding.astype(np.float32).tobytes()
                metadata_json = json.dumps(metadata)
                
                # Insert or update embedding
                cursor.execute("""
                INSERT OR REPLACE INTO embeddings (
                    id, embedding, metadata, created_at, last_accessed
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (embedding_id, embedding_bytes, metadata_json))
                
                conn.commit()
                
                # Update cache size
                with self._stats_lock:
                    cursor.execute("SELECT COUNT(*) FROM embeddings")
                    self.stats['size'] = cursor.fetchone()[0]
                    
        except Exception as e:
            self.logger.error(f"Error storing in cache: {str(e)}")

    def clear_old_entries(self, days_old: int = 30):
        """Clear old cache entries."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                DELETE FROM embeddings 
                WHERE last_accessed < datetime('now', ?)
                """, (f'-{days_old} days',))
                
                conn.commit()
                
                # Update cache size
                with self._stats_lock:
                    cursor.execute("SELECT COUNT(*) FROM embeddings")
                    self.stats['size'] = cursor.fetchone()[0]
                    
        except Exception as e:
            self.logger.error(f"Error clearing old entries: {str(e)}")

    def _generate_id(self, text: str) -> str:
        """Generate cache ID for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._stats_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (
                self.stats['hits'] / total_requests 
                if total_requests > 0 else 0
            )
            
            return {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'size': self.stats['size'],
                'hit_rate': hit_rate
            }

class EmbeddingManager:
    """Manages embedding generation and caching."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize embedding model
        model_name = config.model_config.get(
            'embedding_model',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.model = SentenceTransformer(model_name)
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize cache
        cache_path = config.model_config.get(
            'embedding_cache_path',
            'data/embedding_cache.db'
        )
        self.cache = EmbeddingCache(cache_path)
        
        # Initialize job queue
        self.job_queue = Queue()
        self.processing_threads = config.model_config.get('embedding_threads', 2)
        
        # Start processing threads
        self._start_processing_threads()
        
        # Batch processing settings
        self.batch_size = config.model_config.get('embedding_batch_size', 32)
        self.batch_timeout = config.model_config.get('batch_timeout', 0.1)
        
        # Initialize thread pool for batch processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.processing_threads
        )

    def get_embedding(self, text: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        # Check cache first
        cache_result = self.cache.get(text)
        if cache_result is not None:
            embedding, _ = cache_result
            return embedding
        
        # Generate embedding
        embedding = self._generate_embedding(text)
        
        # Store in cache
        self.cache.put(text, embedding, metadata or {})
        
        return embedding

    def get_embeddings_batch(self, texts: List[str],
                           metadata: Optional[List[Dict[str, Any]]] = None,
                           batch_size: Optional[int] = None) -> List[np.ndarray]:
        """Get embeddings for multiple texts in batches."""
        if not texts:
            return []
            
        batch_size = batch_size or self.batch_size
        results = []
        metadata = metadata or [{}] * len(texts)
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            
            # Check cache for each text
            batch_results = []
            texts_to_generate = []
            texts_to_generate_idx = []
            
            for j, text in enumerate(batch_texts):
                cache_result = self.cache.get(text)
                if cache_result is not None:
                    embedding, _ = cache_result
                    batch_results.append(embedding)
                else:
                    texts_to_generate.append(text)
                    texts_to_generate_idx.append(j)
                    batch_results.append(None)
            
            # Generate missing embeddings
            if texts_to_generate:
                generated_embeddings = self._generate_embeddings_batch(
                    texts_to_generate
                )
                
                # Store in cache and update results
                for idx, embedding in zip(texts_to_generate_idx, generated_embeddings):
                    text = batch_texts[idx]
                    self.cache.put(text, embedding, batch_metadata[idx])
                    batch_results[idx] = embedding
            
            results.extend(batch_results)
        
        return results

    async def get_embedding_async(self, text: str,
                                metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Asynchronously get embedding for text."""
        # Create embedding job
        job = EmbeddingJob(
            id=self.cache._generate_id(text),
            text=text,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        # Add to queue
        self.job_queue.put(job)
        
        # Wait for result
        while True:
            # Check cache
            cache_result = self.cache.get(text)
            if cache_result is not None:
                embedding, _ = cache_result
                return embedding
                
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        with torch.no_grad():
            embedding = self.model.encode(
                text,
                convert_to_tensor=True,
                device=self.device
            )
            return embedding.cpu().numpy()

    def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts."""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                batch_size=self.batch_size
            )
            return embeddings.cpu().numpy()

    def _start_processing_threads(self):
        """Start threads for processing embedding jobs."""
        def processing_task():
            while True:
                try:
                    # Collect batch of jobs
                    batch_jobs = []
                    try:
                        while len(batch_jobs) < self.batch_size:
                            # Get job with timeout
                            job = self.job_queue.get(timeout=self.batch_timeout)
                            batch_jobs.append(job)
                    except Queue.Empty:
                        pass
                    
                    if not batch_jobs:
                        continue
                    
                    # Process batch
                    texts = [job.text for job in batch_jobs]
                    embeddings = self._generate_embeddings_batch(texts)
                    
                    # Store results in cache
                    for job, embedding in zip(batch_jobs, embeddings):
                        self.cache.put(job.text, embedding, job.metadata)
                        
                except Exception as e:
                    self.logger.error(f"Error in embedding processing: {str(e)}")
                    
        for _ in range(self.processing_threads):
            thread = threading.Thread(target=processing_task, daemon=True)
            thread.start()

    def clear_cache(self, days_old: Optional[int] = None):
        """Clear embedding cache."""
        if days_old is not None:
            self.cache.clear_old_entries(days_old)
        else:
            self.cache = EmbeddingCache(self.cache.cache_path)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()