# services/graph_processing.py
from typing import Dict, Any, List, Optional, Set, Union, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict
import numpy as np
import community
import json
import threading
from queue import PriorityQueue
import asyncio
import time

from ..config import Config
from ..database.graph import Neo4jConnection
from .embedding_manager import EmbeddingManager

@dataclass
class GraphSummary:
    """Container for graph summary at different levels."""
    level: str  # 'node', 'community', 'global'
    content: str
    entities: Set[str]
    relationships: Set[tuple]
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class CompressionStats:
    """Statistics for graph compression."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    node_reduction: int
    edge_reduction: int
    execution_time: float

class GraphProcessor:
    """Handles advanced graph processing operations."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection,
                 embedding_manager: EmbeddingManager):
        self.config = config
        self.neo4j = neo4j_connection
        self.embedding_manager = embedding_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize processing configurations
        self.processing_config = config.model_config.get('graph_processing', {})
        
        # Statistics tracking
        self.stats = defaultdict(list)
        self._stats_lock = threading.Lock()
        
        # Initialize cleanup queue
        self.cleanup_queue = PriorityQueue()
        self._start_cleanup_worker()

    async def generate_hierarchical_summary(self, max_levels: int = 3) -> Dict[str, GraphSummary]:
        """Generate hierarchical summary of the graph."""
        try:
            summaries = {}
            
            # 1. Node-level summaries
            node_summaries = await self._generate_node_summaries()
            summaries.update(node_summaries)
            
            # 2. Community-level summaries
            communities = self._detect_communities()
            community_summaries = await self._generate_community_summaries(
                communities,
                node_summaries
            )
            summaries.update(community_summaries)
            
            # 3. Global summary
            global_summary = await self._generate_global_summary(
                communities,
                community_summaries
            )
            summaries['global'] = global_summary
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error generating hierarchical summary: {str(e)}")
            raise

    async def compress_graph(self, compression_level: float = 0.5) -> CompressionStats:
        """Compress graph while preserving important structures."""
        try:
            start_time = time.time()
            
            # Get original graph statistics
            original_stats = self._get_graph_stats()
            
            # 1. Identify redundant structures
            redundant_nodes = self._identify_redundant_nodes(compression_level)
            redundant_edges = self._identify_redundant_edges(compression_level)
            
            # 2. Merge similar nodes
            merged_nodes = await self._merge_similar_nodes(redundant_nodes)
            
            # 3. Remove redundant edges
            removed_edges = self._remove_redundant_edges(redundant_edges)
            
            # 4. Update graph structure
            await self._update_graph_structure(merged_nodes, removed_edges)
            
            # Get compressed graph statistics
            compressed_stats = self._get_graph_stats()
            
            execution_time = time.time() - start_time
            
            return CompressionStats(
                original_size=original_stats['size'],
                compressed_size=compressed_stats['size'],
                compression_ratio=compressed_stats['size'] / original_stats['size'],
                node_reduction=len(merged_nodes),
                edge_reduction=len(removed_edges),
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error compressing graph: {str(e)}")
            raise

    def schedule_cleanup(self, priority: int = 0):
        """Schedule graph cleanup task."""
        cleanup_task = {
            'timestamp': datetime.now(),
            'priority': priority
        }
        self.cleanup_queue.put((priority, cleanup_task))

    async def _generate_node_summaries(self) -> Dict[str, GraphSummary]:
        """Generate summaries for individual nodes."""
        summaries = {}
        
        # Get nodes with their contexts
        nodes = await self._get_nodes_with_context()
        
        for node_id, context in nodes.items():
            try:
                # Generate embedding for node context
                context_embedding = await self.embedding_manager.get_embedding(
                    json.dumps(context)
                )
                
                # Generate summary using embeddings
                summary_text = await self._generate_summary_from_embedding(
                    context_embedding
                )
                
                # Create summary object
                summary = GraphSummary(
                    level='node',
                    content=summary_text,
                    entities={node_id},
                    relationships=set(context['relationships']),
                    metadata={'context_size': len(context['relationships'])},
                    created_at=datetime.now()
                )
                
                summaries[f"node_{node_id}"] = summary
                
            except Exception as e:
                self.logger.error(f"Error summarizing node {node_id}: {str(e)}")
                continue
        
        return summaries

    def _detect_communities(self) -> List[Set[str]]:
        """Detect communities in the graph."""
        try:
            # Get graph data
            G = self._get_networkx_graph()
            
            # Detect communities using Louvain method
            communities = community.best_partition(G)
            
            # Group nodes by community
            community_groups = defaultdict(set)
            for node, comm_id in communities.items():
                community_groups[comm_id].add(node)
            
            return list(community_groups.values())
            
        except Exception as e:
            self.logger.error(f"Error detecting communities: {str(e)}")
            return []

    async def _generate_community_summaries(self, communities: List[Set[str]],
                                         node_summaries: Dict[str, GraphSummary]) -> Dict[str, GraphSummary]:
        """Generate summaries for communities."""
        summaries = {}
        
        for i, community in enumerate(communities):
            try:
                # Get community subgraph
                subgraph = self._get_community_subgraph(community)
                
                # Get community statistics
                stats = self._calculate_community_stats(subgraph)
                
                # Generate community embedding
                community_embedding = await self._generate_community_embedding(
                    subgraph,
                    node_summaries
                )
                
                # Generate summary
                summary_text = await self._generate_summary_from_embedding(
                    community_embedding
                )
                
                # Create summary object
                summary = GraphSummary(
                    level='community',
                    content=summary_text,
                    entities=community,
                    relationships=self._get_community_relationships(subgraph),
                    metadata={
                        'size': len(community),
                        'density': stats['density'],
                        'modularity': stats['modularity']
                    },
                    created_at=datetime.now()
                )
                
                summaries[f"community_{i}"] = summary
                
            except Exception as e:
                self.logger.error(f"Error summarizing community {i}: {str(e)}")
                continue
        
        return summaries

    async def _generate_global_summary(self, communities: List[Set[str]],
                                    community_summaries: Dict[str, GraphSummary]) -> GraphSummary:
        """Generate global graph summary."""
        try:
            # Get global statistics
            global_stats = self._calculate_global_stats()
            
            # Combine community embeddings
            global_embedding = await self._generate_global_embedding(
                community_summaries
            )
            
            # Generate summary
            summary_text = await self._generate_summary_from_embedding(
                global_embedding
            )
            
            # Get all entities and relationships
            all_entities = set()
            all_relationships = set()
            for summary in community_summaries.values():
                all_entities.update(summary.entities)
                all_relationships.update(summary.relationships)
            
            return GraphSummary(
                level='global',
                content=summary_text,
                entities=all_entities,
                relationships=all_relationships,
                metadata={
                    'num_communities': len(communities),
                    'total_nodes': len(all_entities),
                    'total_edges': len(all_relationships),
                    'global_density': global_stats['density'],
                    'global_modularity': global_stats['modularity']
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating global summary: {str(e)}")
            raise

    def _identify_redundant_nodes(self, compression_level: float) -> List[str]:
        """Identify redundant nodes for compression."""
        try:
            redundant_nodes = []
            
            # Get node similarities
            similarities = self._calculate_node_similarities()
            
            # Group similar nodes
            groups = self._group_similar_nodes(similarities)
            
            # Select nodes for compression based on level
            threshold = self._calculate_similarity_threshold(compression_level)
            
            for group in groups:
                if len(group) > 1:
                    similarity = self._calculate_group_similarity(group)
                    if similarity >= threshold:
                        # Keep one representative node
                        redundant_nodes.extend(list(group)[1:])
            
            return redundant_nodes
            
        except Exception as e:
            self.logger.error(f"Error identifying redundant nodes: {str(e)}")
            return []

    def _identify_redundant_edges(self, compression_level: float) -> List[tuple]:
        """Identify redundant edges for compression."""
        try:
            redundant_edges = []
            
            # Get edge importance scores
            edge_scores = self._calculate_edge_importance()
            
            # Sort edges by importance
            sorted_edges = sorted(
                edge_scores.items(),
                key=lambda x: x[1]
            )
            
            # Select edges for removal based on compression level
            num_edges = len(sorted_edges)
            num_remove = int(num_edges * compression_level)
            
            redundant_edges = [
                edge for edge, _ in sorted_edges[:num_remove]
            ]
            
            return redundant_edges
            
        except Exception as e:
            self.logger.error(f"Error identifying redundant edges: {str(e)}")
            return []

    async def _merge_similar_nodes(self, nodes: List[str]) -> Dict[str, str]:
        """Merge similar nodes preserving important connections."""
        merged_nodes = {}
        
        try:
            # Group nodes by similarity
            similarity_groups = await self._group_nodes_by_similarity(nodes)
            
            for group in similarity_groups:
                # Select representative node
                representative = self._select_representative_node(group)
                
                # Merge other nodes into representative
                for node in group:
                    if node != representative:
                        await self._merge_node_into(node, representative)
                        merged_nodes[node] = representative
            
            return merged_nodes
            
        except Exception as e:
            self.logger.error(f"Error merging nodes: {str(e)}")
            return merged_nodes

    async def _update_graph_structure(self, merged_nodes: Dict[str, str],
                                   removed_edges: List[tuple]):
        """Update graph structure after compression."""
        try:
            # Update node mappings
            await self._update_node_mappings(merged_nodes)
            
            # Remove redundant edges
            await self._remove_edges(removed_edges)
            
            # Update indices
            await self._update_indices()
            
            # Validate graph consistency
            await self._validate_graph_consistency()
            
        except Exception as e:
            self.logger.error(f"Error updating graph structure: {str(e)}")
            raise

    def _start_cleanup_worker(self):
        """Start worker thread for graph cleanup."""
        async def cleanup_worker():
            while True:
                try:
                    # Get cleanup task
                    _, task = self.cleanup_queue.get()
                    
                    # Execute cleanup
                    await self._execute_cleanup(task)
                    
                    # Mark task as done
                    self.cleanup_queue.task_done()
                    
                except Exception as e:
                    self.logger.error(f"Error in cleanup worker: {str(e)}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        threading.Thread(
            target=lambda: asyncio.run(cleanup_worker()),
            daemon=True
        ).start()

    async def _execute_cleanup(self, task: Dict[str, Any]):
        """Execute graph cleanup task."""
        try:
            # 1. Remove stale data
            await self._remove_stale_data()
            
            # 2. Optimize indices
            await self._optimize_indices()
            
            # 3. Validate relationships
            await self._validate_relationships()
            
            # 4. Update statistics
            await self._update_graph_statistics()
            
        except Exception as e:
            self.logger.error(f"Error executing cleanup: {str(e)}")
            raise

    async def _remove_stale_data(self):
        """Remove stale nodes and relationships."""
        try:
            cutoff_date = datetime.now() - timedelta(
                days=self.processing_config.get('stale_data_days', 90)
            )
            
            query = """
            MATCH (n)
            WHERE n.last_accessed < $cutoff
            WITH n, size((n)--()) as degree
            WHERE degree = 0
            DELETE n
            """
            
            with self.neo4j._driver.session() as session:
                result = await session.run(query, cutoff=cutoff_date)
                self.logger.info(f"Removed {result.summary().counters.nodes_deleted} stale nodes")
                
        except Exception as e:
            self.logger.error(f"Error removing stale data: {str(e)}")
            raise

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get graph processing statistics."""
        with self._stats_lock:
            stats = {}
            
            for stat_name, values in self.stats.items():
                if values:
                    stats[stat_name] = {
                        'count': len(values),
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            return stats