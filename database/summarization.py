# database/summarization.py
from typing import List, Dict, Any, Optional, Set
from typing import Tuple 
import logging
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
from collections import defaultdict
import torch
import json

#from ..config import Config
from config import Config
from .community import Community
from .graph import Neo4jConnection

@dataclass
class GraphSummary:
    """Container for graph summary at different levels."""
    id: str
    level: str  # 'node', 'community', 'global'
    content: str
    metadata: Dict[str, Any]
    entities: Set[str]
    relationships: Set[Tuple[str, str, str]]
    created_at: datetime
    last_updated: datetime

class GraphSummarizer:
    """Handles multi-level graph summarization."""
    
    def __init__(self, config: Config, neo4j_connection: Neo4jConnection):
        self.config = config
        self.neo4j = neo4j_connection
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Summary configurations
        self.summary_configs = {
            'node': {
                'max_context_nodes': 10,
                'max_relationships': 20,
                'include_attributes': True
            },
            'community': {
                'min_size': 3,
                'max_summary_length': 500,
                'include_statistics': True
            },
            'global': {
                'max_communities': 10,
                'max_summary_length': 1000,
                'include_trends': True
            }
        }

    def generate_summaries(self, communities: List[Community], 
                         llm_processor) -> Dict[str, GraphSummary]:
        """Generate summaries at all levels."""
        try:
            summaries = {}
            
            # 1. Node-level summaries
            node_summaries = self._generate_node_summaries(communities, llm_processor)
            summaries.update(node_summaries)
            
            # 2. Community-level summaries
            community_summaries = self._generate_community_summaries(
                communities,
                node_summaries,
                llm_processor
            )
            summaries.update(community_summaries)
            
            # 3. Global summary
            global_summary = self._generate_global_summary(
                communities,
                community_summaries,
                llm_processor
            )
            summaries['global'] = global_summary
            
            # Store summaries in Neo4j
            self._store_summaries(summaries)
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error generating summaries: {str(e)}")
            raise

    def _generate_node_summaries(self, communities: List[Community],
                               llm_processor) -> Dict[str, GraphSummary]:
        """Generate node-level summaries."""
        config = self.summary_configs['node']
        summaries = {}
        
        for community in communities:
            for node_id in community.nodes:
                try:
                    # Get node context
                    context = self._get_node_context(
                        node_id,
                        config['max_context_nodes']
                    )
                    
                    # Generate summary using LLM
                    summary_text = self._generate_node_summary(
                        context,
                        llm_processor
                    )
                    
                    # Create summary object
                    summary = GraphSummary(
                        id=f"node_{node_id}",
                        level='node',
                        content=summary_text,
                        metadata=self._get_node_metadata(node_id),
                        entities={node_id},
                        relationships=set(context['relationships']),
                        created_at=datetime.now(),
                        last_updated=datetime.now()
                    )
                    
                    summaries[summary.id] = summary
                    
                except Exception as e:
                    self.logger.error(f"Error summarizing node {node_id}: {str(e)}")
                    continue
        
        return summaries

    def _generate_community_summaries(self, communities: List[Community],
                                    node_summaries: Dict[str, GraphSummary],
                                    llm_processor) -> Dict[str, GraphSummary]:
        """Generate community-level summaries."""
        config = self.summary_configs['community']
        summaries = {}
        
        for community in communities:
            if len(community.nodes) < config['min_size']:
                continue
                
            try:
                # Collect node summaries for this community
                community_node_summaries = [
                    node_summaries[f"node_{node_id}"]
                    for node_id in community.nodes
                    if f"node_{node_id}" in node_summaries
                ]
                
                # Get community statistics
                stats = self._get_community_statistics(community)
                
                # Generate community summary
                summary_text = self._generate_community_summary(
                    community_node_summaries,
                    stats,
                    llm_processor
                )
                
                # Create summary object
                summary = GraphSummary(
                    id=f"community_{community.id}",
                    level='community',
                    content=summary_text,
                    metadata={
                        'statistics': stats,
                        'size': len(community.nodes)
                    },
                    entities=community.nodes,
                    relationships=self._get_community_relationships(community),
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                
                summaries[summary.id] = summary
                
            except Exception as e:
                self.logger.error(
                    f"Error summarizing community {community.id}: {str(e)}"
                )
                continue
        
        return summaries

    def _generate_global_summary(self, communities: List[Community],
                               community_summaries: Dict[str, GraphSummary],
                               llm_processor) -> GraphSummary:
        """Generate global graph summary."""
        config = self.summary_configs['global']
        
        try:
            # Select top communities by size
            top_communities = sorted(
                communities,
                key=lambda c: len(c.nodes),
                reverse=True
            )[:config['max_communities']]
            
            # Get global statistics
            global_stats = self._get_global_statistics(communities)
            
            # Collect community summaries
            relevant_summaries = [
                community_summaries[f"community_{c.id}"]
                for c in top_communities
                if f"community_{c.id}" in community_summaries
            ]
            
            # Generate global summary
            summary_text = self._generate_global_summary_text(
                relevant_summaries,
                global_stats,
                llm_processor
            )
            
            # Create summary object
            all_entities = set()
            all_relationships = set()
            for summary in community_summaries.values():
                all_entities.update(summary.entities)
                all_relationships.update(summary.relationships)
            
            return GraphSummary(
                id="global_summary",
                level='global',
                content=summary_text,
                metadata={
                    'statistics': global_stats,
                    'community_count': len(communities)
                },
                entities=all_entities,
                relationships=all_relationships,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating global summary: {str(e)}")
            raise

    def _get_node_context(self, node_id: str, 
                         max_context: int) -> Dict[str, Any]:
        """Get context information for a node."""
        query = """
        MATCH (n:Entity {id: $node_id})-[r]-(m:Entity)
        RETURN n.text as source_text,
               n.type as source_type,
               type(r) as relationship,
               m.id as target_id,
               m.text as target_text,
               m.type as target_type
        LIMIT $max_context
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_id=node_id, max_context=max_context)
            
            context = {
                'central_node': node_id,
                'relationships': [],
                'connected_nodes': {}
            }
            
            for record in result:
                # Add relationship
                context['relationships'].append((
                    record['source_text'],
                    record['relationship'],
                    record['target_text']
                ))
                
                # Add connected node
                context['connected_nodes'][record['target_id']] = {
                    'text': record['target_text'],
                    'type': record['target_type']
                }
            
            return context

    def _generate_node_summary(self, context: Dict[str, Any],
                             llm_processor) -> str:
        """Generate summary for a node using LLM."""
        # Format context for LLM
        prompt = self._format_node_summary_prompt(context)
        
        # Generate summary using LLM
        summary = llm_processor.generate_text(prompt)
        
        return summary

    def _format_node_summary_prompt(self, context: Dict[str, Any]) -> str:
        """Format node context into LLM prompt."""
        prompt = "Generate a concise summary of the following entity and its relationships:\n\n"
        
        # Add relationships
        prompt += "Relationships:\n"
        for source, rel, target in context['relationships']:
            prompt += f"- {source} {rel} {target}\n"
        
        prompt += "\nProvide a natural language summary that describes the entity's key relationships and role in the network."
        
        return prompt

    def _get_node_metadata(self, node_id: str) -> Dict[str, Any]:
        """Get metadata for a node."""
        query = """
        MATCH (n:Entity {id: $node_id})
        RETURN properties(n) as props,
               size((n)-[]->()) as out_degree,
               size((n)<-[]-()) as in_degree
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            
            metadata = record['props']
            metadata.update({
                'out_degree': record['out_degree'],
                'in_degree': record['in_degree']
            })
            
            return metadata

    def _get_community_statistics(self, community: Community) -> Dict[str, Any]:
        """Calculate statistics for a community."""
        query = """
        MATCH (n:Entity)
        WHERE n.id IN $node_ids
        WITH collect(n) as nodes
        RETURN size(nodes) as node_count,
               size([x in nodes where x.type = 'PERSON']) as person_count,
               size([x in nodes where x.type = 'ORGANIZATION']) as org_count,
               size([x in nodes where x.type = 'LOCATION']) as location_count,
               size([x in nodes where x.type = 'EVENT']) as event_count,
               size((nodes)-[]-(nodes)) as internal_edges
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_ids=list(community.nodes))
            record = result.single()
            
            return {
                'node_count': record['node_count'],
                'entity_types': {
                    'PERSON': record['person_count'],
                    'ORGANIZATION': record['org_count'],
                    'LOCATION': record['location_count'],
                    'EVENT': record['event_count']
                },
                'internal_edges': record['internal_edges'],
                'density': record['internal_edges'] / (
                    record['node_count'] * (record['node_count'] - 1)
                ) if record['node_count'] > 1 else 0
            }

    def _generate_community_summary(self, 
                                  node_summaries: List[GraphSummary],
                                  stats: Dict[str, Any],
                                  llm_processor) -> str:
        """Generate summary for a community using LLM."""
        # Format community information for LLM
        prompt = self._format_community_summary_prompt(node_summaries, stats)
        
        # Generate summary using LLM
        summary = llm_processor.generate_text(prompt)
        
        return summary

    def _format_community_summary_prompt(self, 
                                       node_summaries: List[GraphSummary],
                                       stats: Dict[str, Any]) -> str:
        """Format community information into LLM prompt."""
        prompt = "Generate a comprehensive summary of the following community:\n\n"
        
        # Add statistics
        prompt += "Statistics:\n"
        prompt += f"- Total nodes: {stats['node_count']}\n"
        prompt += "- Entity types:\n"
        for type_name, count in stats['entity_types'].items():
            if count > 0:
                prompt += f"  * {type_name}: {count}\n"
        prompt += f"- Network density: {stats['density']:.2f}\n\n"
        
        # Add node summaries
        prompt += "Key entities and their relationships:\n"
        for summary in node_summaries:
            prompt += f"- {summary.content}\n"
        
        prompt += "\nProvide a natural language summary that describes the community's composition, key entities, and notable patterns."
        
        return prompt

    def _get_community_relationships(self, community: Community) -> Set[Tuple[str, str, str]]:
        """Get all relationships within a community."""
        query = """
        MATCH (n:Entity)-[r]->(m:Entity)
        WHERE n.id IN $node_ids AND m.id IN $node_ids
        RETURN n.id as source, type(r) as rel_type, m.id as target
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, node_ids=list(community.nodes))
            return {(r['source'], r['rel_type'], r['target']) for r in result}

    def _get_global_statistics(self, communities: List[Community]) -> Dict[str, Any]:
        """Calculate global statistics."""
        query = """
        MATCH (n:Entity)
        WITH count(n) as total_nodes
        MATCH ()-[r]->()
        WITH total_nodes, count(r) as total_edges
        RETURN total_nodes, total_edges
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            record = result.single()
            
            return {
                'total_nodes': record['total_nodes'],
                'total_edges': record['total_edges'],
                'total_communities': len(communities),
                'avg_community_size': sum(
                    len(c.nodes) for c in communities
                ) / len(communities) if communities else 0,
                'largest_community_size': max(
                    len(c.nodes) for c in communities
                ) if communities else 0
            }

    def _generate_global_summary_text(self, community_summaries: List[GraphSummary],
                                   global_stats: Dict[str, Any],
                                   llm_processor) -> str:
        """Generate global summary text using LLM."""
        # Format global information for LLM
        prompt = self._format_global_summary_prompt(
            community_summaries,
            global_stats
        )
        
        # Generate summary using LLM
        summary = llm_processor.generate_text(prompt)
        
        return summary

    def _format_global_summary_prompt(self, community_summaries: List[GraphSummary],
                                    global_stats: Dict[str, Any]) -> str:
        """Format global information into LLM prompt."""
        prompt = "Generate a comprehensive summary of the entire knowledge graph:\n\n"
        
        # Add global statistics
        prompt += "Global Statistics:\n"
        prompt += f"- Total nodes: {global_stats['total_nodes']}\n"
        prompt += f"- Total edges: {global_stats['total_edges']}\n"
        prompt += f"- Total communities: {global_stats['total_communities']}\n"
        prompt += f"- Average community size: {global_stats['avg_community_size']:.1f}\n"
        prompt += f"- Largest community size: {global_stats['largest_community_size']}\n\n"
        
        # Add community summaries
        prompt += "Key Communities:\n"
        for summary in community_summaries:
            prompt += f"Community: {summary.content}\n\n"
        
        prompt += "\nProvide a high-level summary that describes the overall structure, major communities, and key patterns in the knowledge graph."
        
        return prompt

    def _store_summaries(self, summaries: Dict[str, GraphSummary]):
        """Store summaries in Neo4j."""
        for summary_id, summary in summaries.items():
            try:
                self._store_single_summary(summary)
            except Exception as e:
                self.logger.error(
                    f"Error storing summary {summary_id}: {str(e)}"
                )
                continue

    def _store_single_summary(self, summary: GraphSummary):
        """Store a single summary in Neo4j."""
        query = """
        MERGE (s:Summary {id: $id})
        SET s.level = $level,
            s.content = $content,
            s.metadata = $metadata,
            s.created_at = datetime($created_at),
            s.last_updated = datetime($last_updated)
        WITH s
        UNWIND $entities as entity_id
        MATCH (e:Entity {id: entity_id})
        MERGE (e)-[:HAS_SUMMARY]->(s)
        """
        
        with self.neo4j._driver.session() as session:
            session.run(
                query,
                id=summary.id,
                level=summary.level,
                content=summary.content,
                metadata=summary.metadata,
                created_at=summary.created_at.isoformat(),
                last_updated=summary.last_updated.isoformat(),
                entities=list(summary.entities)
            )

    def get_summary(self, summary_id: str) -> Optional[GraphSummary]:
        """Retrieve a specific summary."""
        query = """
        MATCH (s:Summary {id: $id})
        OPTIONAL MATCH (e:Entity)-[:HAS_SUMMARY]->(s)
        WITH s, collect(e.id) as entity_ids
        RETURN s.level as level,
               s.content as content,
               s.metadata as metadata,
               entity_ids,
               s.created_at as created_at,
               s.last_updated as last_updated
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, id=summary_id)
            record = result.single()
            
            if not record:
                return None
            
            return GraphSummary(
                id=summary_id,
                level=record['level'],
                content=record['content'],
                metadata=record['metadata'],
                entities=set(record['entity_ids']),
                relationships=set(),  # Would need additional query to get relationships
                created_at=record['created_at'],
                last_updated=record['last_updated']
            )

    def get_summaries_by_level(self, level: str) -> List[GraphSummary]:
        """Retrieve all summaries at a specific level."""
        query = """
        MATCH (s:Summary {level: $level})
        OPTIONAL MATCH (e:Entity)-[:HAS_SUMMARY]->(s)
        WITH s, collect(e.id) as entity_ids
        RETURN s.id as id,
               s.content as content,
               s.metadata as metadata,
               entity_ids,
               s.created_at as created_at,
               s.last_updated as last_updated
        """
        
        with self.neo4j._driver.session() as session:
            result = session.run(query, level=level)
            
            summaries = []
            for record in result:
                summary = GraphSummary(
                    id=record['id'],
                    level=level,
                    content=record['content'],
                    metadata=record['metadata'],
                    entities=set(record['entity_ids']),
                    relationships=set(),
                    created_at=record['created_at'],
                    last_updated=record['last_updated']
                )
                summaries.append(summary)
            
            return summaries