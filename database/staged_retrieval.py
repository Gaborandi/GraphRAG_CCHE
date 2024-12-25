# database/staged_retrieval.py
from typing import List, Dict, Any, Optional, Set
import logging
from dataclasses import dataclass
from datetime import datetime
import torch
from torch import Tensor
import numpy as np
from enum import Enum

from ..config import Config
from .community import Community, CommunityDetector
from .retrieval import GraphRetriever, RetrievalResult
from .graph import Neo4jConnection

class RetrievalStage(Enum):
    """Enum for different retrieval stages."""
    COMMUNITY_SELECTION = "community_selection"
    INITIAL_RETRIEVAL = "initial_retrieval"
    CONTEXT_EXPANSION = "context_expansion"
    RELEVANCE_REFINEMENT = "relevance_refinement"
    FINAL_RANKING = "final_ranking"

@dataclass
class StageMetrics:
    """Metrics for each retrieval stage."""
    stage: RetrievalStage
    duration: float
    nodes_processed: int
    edges_processed: int
    memory_usage: float
    success_rate: float

@dataclass
class StagedRetrievalResult:
    """Results from staged retrieval process."""
    final_results: RetrievalResult
    stage_metrics: List[StageMetrics]
    total_time: float
    overall_confidence: float

class StagedRetriever:
    """Implements multi-stage retrieval process."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.community_detector = CommunityDetector(config, neo4j_connection)
        self.graph_retriever = GraphRetriever(config, neo4j_connection)
        
        # Stage-specific configurations
        self.stage_configs = {
            RetrievalStage.COMMUNITY_SELECTION: {
                'max_communities': 5,
                'min_similarity': 0.3
            },
            RetrievalStage.INITIAL_RETRIEVAL: {
                'max_nodes': 100,
                'max_depth': 2
            },
            RetrievalStage.CONTEXT_EXPANSION: {
                'expansion_factor': 1.5,
                'max_additional_nodes': 50
            },
            RetrievalStage.RELEVANCE_REFINEMENT: {
                'relevance_threshold': 0.7,
                'min_connections': 2
            },
            RetrievalStage.FINAL_RANKING: {
                'top_k': 10,
                'diversity_weight': 0.3
            }
        }

    def staged_retrieve(self, query: str, embedding_model) -> StagedRetrievalResult:
        """
        Perform multi-stage retrieval process.
        Args:
            query: Search query
            embedding_model: Model for text embeddings
        """
        start_time = datetime.now()
        stage_metrics = []
        
        try:
            # Stage 1: Community Selection
            stage_start = datetime.now()
            relevant_communities = self._select_communities(query, embedding_model)
            stage_metrics.append(self._calculate_stage_metrics(
                RetrievalStage.COMMUNITY_SELECTION,
                stage_start,
                len(relevant_communities)
            ))
            
            # Stage 2: Initial Retrieval
            stage_start = datetime.now()
            initial_results = self._perform_initial_retrieval(
                query,
                relevant_communities,
                embedding_model
            )
            stage_metrics.append(self._calculate_stage_metrics(
                RetrievalStage.INITIAL_RETRIEVAL,
                stage_start,
                self._count_nodes_edges(initial_results)
            ))
            
            # Stage 3: Context Expansion
            stage_start = datetime.now()
            expanded_results = self._expand_context(
                query,
                initial_results,
                embedding_model
            )
            stage_metrics.append(self._calculate_stage_metrics(
                RetrievalStage.CONTEXT_EXPANSION,
                stage_start,
                self._count_nodes_edges(expanded_results)
            ))
            
            # Stage 4: Relevance Refinement
            stage_start = datetime.now()
            refined_results = self._refine_relevance(
                query,
                expanded_results,
                embedding_model
            )
            stage_metrics.append(self._calculate_stage_metrics(
                RetrievalStage.RELEVANCE_REFINEMENT,
                stage_start,
                self._count_nodes_edges(refined_results)
            ))
            
            # Stage 5: Final Ranking
            stage_start = datetime.now()
            final_results = self._perform_final_ranking(
                query,
                refined_results,
                embedding_model
            )
            stage_metrics.append(self._calculate_stage_metrics(
                RetrievalStage.FINAL_RANKING,
                stage_start,
                self._count_nodes_edges(final_results)
            ))
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            return StagedRetrievalResult(
                final_results=final_results,
                stage_metrics=stage_metrics,
                total_time=total_time,
                overall_confidence=self._calculate_overall_confidence(
                    final_results,
                    stage_metrics
                )
            )
            
        except Exception as e:
            self.logger.error(f"Error in staged retrieval: {str(e)}")
            raise

    def _select_communities(self, query: str, 
                          embedding_model) -> List[Community]:
        """Stage 1: Select relevant communities."""
        config = self.stage_configs[RetrievalStage.COMMUNITY_SELECTION]
        
        # Get query embedding
        query_embedding = self.graph_retriever._get_query_embedding(
            query,
            embedding_model
        )
        
        # Get communities and their relevance scores
        communities = self.community_detector.detect_communities()
        community_scores = []
        
        for community in communities:
            if community.summary:
                summary_embedding = self._get_text_embedding(
                    community.summary,
                    embedding_model
                )
                similarity = torch.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    summary_embedding.unsqueeze(0)
                ).item()
                
                if similarity >= config['min_similarity']:
                    community_scores.append((similarity, community))
        
        # Sort and select top communities
        community_scores.sort(reverse=True)
        return [c for _, c in community_scores[:config['max_communities']]]

    def _perform_initial_retrieval(self, query: str,
                                 communities: List[Community],
                                 embedding_model) -> RetrievalResult:
        """Stage 2: Perform initial retrieval within selected communities."""
        config = self.stage_configs[RetrievalStage.INITIAL_RETRIEVAL]
        
        # Collect all nodes from selected communities
        community_nodes = set()
        for community in communities:
            community_nodes.update(community.nodes)
        
        # Perform retrieval within community nodes
        return self.graph_retriever.retrieve(
            query,
            embedding_model,
            max_subgraphs=config['max_nodes'],
            max_depth=config['max_depth'],
            restricted_nodes=community_nodes
        )

    def _expand_context(self, query: str,
                       initial_results: RetrievalResult,
                       embedding_model) -> RetrievalResult:
        """Stage 3: Expand context around initial results."""
        config = self.stage_configs[RetrievalStage.CONTEXT_EXPANSION]
        
        # Get existing nodes
        existing_nodes = set()
        for subgraph in initial_results.subgraphs:
            existing_nodes.update(subgraph['nodes'])
        
        # Find additional related nodes
        additional_nodes = self._find_related_nodes(
            existing_nodes,
            config['max_additional_nodes']
        )
        
        # Merge results
        expanded_subgraphs = self._merge_subgraphs(
            initial_results.subgraphs,
            additional_nodes
        )
        
        return RetrievalResult(
            subgraphs=expanded_subgraphs,
            communities=initial_results.communities,
            search_path=initial_results.search_path,
            confidence=initial_results.confidence,
            retrieval_time=initial_results.retrieval_time
        )

    def _refine_relevance(self, query: str,
                         expanded_results: RetrievalResult,
                         embedding_model) -> RetrievalResult:
        """Stage 4: Refine results based on relevance."""
        config = self.stage_configs[RetrievalStage.RELEVANCE_REFINEMENT]
        
        # Get query embedding
        query_embedding = self.graph_retriever._get_query_embedding(
            query,
            embedding_model
        )
        
        refined_subgraphs = []
        for subgraph in expanded_results.subgraphs:
            # Calculate subgraph relevance
            relevance = self._calculate_subgraph_relevance(
                subgraph,
                query_embedding,
                embedding_model
            )
            
            if relevance >= config['relevance_threshold']:
                refined_subgraphs.append(subgraph)
        
        return RetrievalResult(
            subgraphs=refined_subgraphs,
            communities=expanded_results.communities,
            search_path=expanded_results.search_path,
            confidence=expanded_results.confidence,
            retrieval_time=expanded_results.retrieval_time
        )

    def _perform_final_ranking(self, query: str,
                             refined_results: RetrievalResult,
                             embedding_model) -> RetrievalResult:
        """Stage 5: Final ranking of results."""
        config = self.stage_configs[RetrievalStage.FINAL_RANKING]
        
        # Score subgraphs based on relevance and diversity
        scored_subgraphs = []
        for subgraph in refined_results.subgraphs:
            relevance_score = self._calculate_subgraph_relevance(
                subgraph,
                query_embedding,
                embedding_model
            )
            diversity_score = self._calculate_diversity_score(
                subgraph,
                scored_subgraphs
            )
            
            final_score = (
                (1 - config['diversity_weight']) * relevance_score +
                config['diversity_weight'] * diversity_score
            )
            
            scored_subgraphs.append((final_score, subgraph))
        
        # Sort and select top-k
        scored_subgraphs.sort(reverse=True)
        final_subgraphs = [sg for _, sg in scored_subgraphs[:config['top_k']]]
        
        return RetrievalResult(
            subgraphs=final_subgraphs,
            communities=refined_results.communities,
            search_path=refined_results.search_path,
            confidence=refined_results.confidence,
            retrieval_time=refined_results.retrieval_time
        )

    def _calculate_stage_metrics(self, stage: RetrievalStage,
                               start_time: datetime,
                               nodes_processed: int) -> StageMetrics:
        """Calculate metrics for a retrieval stage."""
        duration = (datetime.now() - start_time).total_seconds()
        
        return StageMetrics(
            stage=stage,
            duration=duration,
            nodes_processed=nodes_processed,
            edges_processed=self._estimate_edges_processed(nodes_processed),
            memory_usage=self._get_memory_usage(),
            success_rate=self._calculate_success_rate(stage)
        )

    def _calculate_overall_confidence(self, 
                                   final_results: RetrievalResult,
                                   stage_metrics: List[StageMetrics]) -> float:
        """Calculate overall confidence score."""
        # Combine multiple factors
        result_confidence = final_results.confidence
        stage_success = np.mean([m.success_rate for m in stage_metrics])
        
        return (result_confidence + stage_success) / 2

    def _find_related_nodes(self, seed_nodes: Set[str],
                          max_nodes: int) -> Set[str]:
        """Find related nodes based on graph structure."""
        query = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE n.id IN $seed_nodes
        AND NOT m.id IN $seed_nodes
        WITH m, count(r) as connection_count
        ORDER BY connection_count DESC
        LIMIT $max_nodes
        RETURN m.id
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(
                query,
                seed_nodes=list(seed_nodes),
                max_nodes=max_nodes
            )
            return {record[0] for record in result}

    def _calculate_subgraph_relevance(self, subgraph: Dict[str, Any],
                                    query_embedding: Tensor,
                                    embedding_model) -> float:
        """Calculate relevance score for a subgraph."""
        # Get embeddings for subgraph nodes
        node_embeddings = []
        for node_id in subgraph['nodes']:
            embedding = self._get_node_embedding(node_id, embedding_model)
            node_embeddings.append(embedding)
        
        if not node_embeddings:
            return 0.0
        
        # Calculate average similarity
        similarities = [
            torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                emb.unsqueeze(0)
            ).item()
            for emb in node_embeddings
        ]
        
        return np.mean(similarities)

    def _calculate_diversity_score(self, subgraph: Dict[str, Any],
                                 existing_subgraphs: List[Dict[str, Any]]) -> float:
        """Calculate diversity score based on overlap with existing results."""
        if not existing_subgraphs:
            return 1.0
        
        current_nodes = set(subgraph['nodes'])
        overlaps = []
        
        for existing in existing_subgraphs:
            existing_nodes = set(existing['nodes'])
            overlap = len(current_nodes.intersection(existing_nodes)) / len(current_nodes)
            overlaps.append(overlap)
        
        return 1.0 - np.mean(overlaps)

    @staticmethod
    def _count_nodes_edges(results: RetrievalResult) -> int:
        """Count total nodes processed in results."""
        nodes = set()
        for subgraph in results.subgraphs:
            nodes.update(subgraph['nodes'])
        return len(nodes)

    @staticmethod
    def _estimate_edges_processed(nodes: int) -> int:
        """Estimate number of edges processed based on nodes."""
        # Assuming average degree of 4
        return nodes * 2

    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in GB."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
        except ImportError:
            return 0.0

    def _calculate_success_rate(self, stage: RetrievalStage) -> float:
        """Calculate success rate for a retrieval stage."""
        # Implementation depends on stage-specific success criteria
        stage_criteria = {
            RetrievalStage.COMMUNITY_SELECTION: self._community_selection_success,
            RetrievalStage.INITIAL_RETRIEVAL: self._initial_retrieval_success,
            RetrievalStage.CONTEXT_EXPANSION: self._context_expansion_success,
            RetrievalStage.RELEVANCE_REFINEMENT: self._relevance_refinement_success,
            RetrievalStage.FINAL_RANKING: self._final_ranking_success
        }
        
        if stage in stage_criteria:
            return stage_criteria[stage]()
        return 1.0  # Default success rate

    def _community_selection_success(self) -> float:
        """Calculate success rate for community selection."""
        config = self.stage_configs[RetrievalStage.COMMUNITY_SELECTION]
        try:
            with self.neo4j._driver.session() as session:
                # Check if selected communities have expected connectivity
                query = """
                MATCH (n:Community)
                WITH count(n) as total_communities
                RETURN total_communities
                """
                result = session.run(query)
                total = result.single()[0]
                if total == 0:
                    return 0.0
                return min(1.0, config['max_communities'] / total)
        except Exception:
            return 0.0

    def _initial_retrieval_success(self) -> float:
        """Calculate success rate for initial retrieval."""
        config = self.stage_configs[RetrievalStage.INITIAL_RETRIEVAL]
        try:
            with self.neo4j._driver.session() as session:
                query = """
                MATCH (n:Entity)
                WITH count(n) as total_nodes
                RETURN total_nodes
                """
                result = session.run(query)
                total = result.single()[0]
                if total == 0:
                    return 0.0
                return min(1.0, config['max_nodes'] / total)
        except Exception:
            return 0.0

    def _context_expansion_success(self) -> float:
        """Calculate success rate for context expansion."""
        # Base success on expansion factor achievement
        config = self.stage_configs[RetrievalStage.CONTEXT_EXPANSION]
        target_expansion = config['expansion_factor']
        try:
            actual_expansion = len(set().union(*[s['nodes'] for s in self.current_results.subgraphs])) / \
                             len(set().union(*[s['nodes'] for s in self.initial_results.subgraphs]))
            return min(1.0, actual_expansion / target_expansion)
        except Exception:
            return 0.0

    def _relevance_refinement_success(self) -> float:
        """Calculate success rate for relevance refinement."""
        config = self.stage_configs[RetrievalStage.RELEVANCE_REFINEMENT]
        try:
            # Calculate what percentage of subgraphs meet relevance threshold
            relevant_count = sum(1 for sg in self.current_results.subgraphs 
                               if self._calculate_subgraph_relevance(sg) >= config['relevance_threshold'])
            total_count = len(self.current_results.subgraphs)
            return relevant_count / total_count if total_count > 0 else 0.0
        except Exception:
            return 0.0

    def _final_ranking_success(self) -> float:
        """Calculate success rate for final ranking."""
        config = self.stage_configs[RetrievalStage.FINAL_RANKING]
        try:
            # Success based on achieving target number of results
            actual_results = len(self.current_results.subgraphs)
            return min(1.0, actual_results / config['top_k'])
        except Exception:
            return 0.0

    def _merge_subgraphs(self, subgraphs: List[Dict[str, Any]], 
                        additional_nodes: Set[str]) -> List[Dict[str, Any]]:
        """Merge subgraphs with additional nodes."""
        merged_subgraphs = []
        
        for subgraph in subgraphs:
            # Create new subgraph with existing nodes
            new_subgraph = {
                'nodes': set(subgraph['nodes']),
                'edges': set(subgraph['edges']),
                'center': subgraph['center']
            }
            
            # Find edges to additional nodes
            query = """
            MATCH (n:Entity)-[r]->(m:Entity)
            WHERE n.id IN $base_nodes AND m.id IN $additional_nodes
            RETURN n.id as source, type(r) as rel_type, m.id as target
            """
            
            with self.neo4j._driver.session() as session:
                result = session.run(
                    query,
                    base_nodes=list(new_subgraph['nodes']),
                    additional_nodes=list(additional_nodes)
                )
                
                # Add new nodes and edges
                for record in result:
                    new_subgraph['nodes'].add(record['target'])
                    new_subgraph['edges'].add((
                        record['source'],
                        record['rel_type'],
                        record['target']
                    ))
            
            merged_subgraphs.append(new_subgraph)
        
        return merged_subgraphs