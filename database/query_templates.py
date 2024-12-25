# database/query_templates.py
class GraphQueries:
    """Collection of common graph queries."""
    
    @staticmethod
    def get_entity_by_id(entity_id: str) -> str:
        return """
        MATCH (e:Entity {id: $id})
        RETURN e
        """
    
    @staticmethod
    def get_entity_relationships(entity_id: str) -> str:
        return """
        MATCH (e:Entity {id: $id})-[r]-(related)
        RETURN type(r) as relationship_type, 
               r.confidence as confidence,
               related
        """
    
    @staticmethod
    def get_document_entities(doc_id: str) -> str:
        return """
        MATCH (d:Document {id: $id})<-[r:MENTIONED_IN]-(e:Entity)
        RETURN e.text as entity_text,
               e.type as entity_type,
               r.mention_count as mentions,
               r.chunks as chunks
        """