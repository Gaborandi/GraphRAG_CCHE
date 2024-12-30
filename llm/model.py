# llm/model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from threading import Lock
import json

#from ..config import Config
#from ..document_processor.base import Document

from config import Config              
from document_processor.base import DocumentProcessor
from document_processor.base import Document

@dataclass
class ExtractionResult:
    """Container for extraction results."""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    confidence: float
    chunk_id: str

class LlamaProcessor:
    """Handles Llama model operations and text processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model_lock = Lock()
        
        self.logger.info(f"Initializing Llama model: {config.model_name}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.batch_processor = BatchProcessor(
            batch_size=config.get('batch_size', 32),
            max_retries=config.get('max_retries', 3)
        )
        self.fallback_model = config.get('fallback_model')
        
        # Load prompt templates
        self.prompts = self._load_prompt_templates()
        
        self.logger.info("Llama model initialization complete")
        
    async def process_batch(self, texts):
                try:
                    return await self.batch_processor.process(
                        texts,
                        self.model,
                        self.tokenizer
                    )
                except Exception as e:
                    if self.fallback_model:
                        return await self._process_with_fallback(texts)
                    raise LLMProcessingError("Batch processing failed") from e
        
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from configuration."""
        return {
            'entity_extraction': """Extract key entities from the following text. Include people, organizations, locations, and dates.
Text: {text}
Entities:""",
            
            'relationship_extraction': """Given the following text and entities, identify relationships between them.
Text: {text}
Entities: {entities}
Relationships:""",
            
            'attribute_extraction': """For each entity, extract relevant attributes and properties.
Entity: {entity}
Context: {context}
Attributes:"""
        }

    def process_document(self, document: Document) -> List[ExtractionResult]:
        """Process a document and extract information."""
        results = []
        
        for chunk in document.chunks:
            try:
                # Process each chunk
                chunk_result = self._process_chunk(chunk['text'], chunk['chunk_id'])
                results.append(chunk_result)
                
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk['chunk_id']}: {str(e)}")
                continue
        
        return results

    def _process_chunk(self, text: str, chunk_id: str) -> ExtractionResult:
        """Process a single text chunk."""
        # Extract entities
        entities = self._extract_entities(text)
        
        # Extract relationships if entities were found
        relationships = []
        if entities:
            relationships = self._extract_relationships(text, entities)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(entities, relationships)
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            confidence=confidence,
            chunk_id=chunk_id
        )

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using Llama model."""
        prompt = self.prompts['entity_extraction'].format(text=text)
        
        with self._model_lock:
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Parse entities from response
                entities = self._parse_entities(response)
                
                return entities
                
            except Exception as e:
                self.logger.error(f"Error in entity extraction: {str(e)}")
                return []

    def _extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        prompt = self.prompts['relationship_extraction'].format(
            text=text,
            entities=json.dumps(entities, indent=2)
        )
        
        with self._model_lock:
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Parse relationships from response
                relationships = self._parse_relationships(response)
                
                return relationships
                
            except Exception as e:
                self.logger.error(f"Error in relationship extraction: {str(e)}")
                return []

    def _parse_entities(self, response: str) -> List[Dict[str, Any]]:
        """Parse entities from model response."""
        entities = []
        try:
            # Split response to get only the generated part
            generated_text = response.split("Entities:")[1].strip()
            
            # Parse the response into structured format
            # This is a simplified implementation - enhance based on actual model output
            lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
            
            for line in lines:
                if ':' in line:
                    entity_type, entity_text = line.split(':', 1)
                    entities.append({
                        'type': entity_type.strip(),
                        'text': entity_text.strip(),
                        'start_idx': -1,  # Would need text matching to find exact indices
                        'end_idx': -1
                    })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error parsing entities: {str(e)}")
            return []

    def _parse_relationships(self, response: str) -> List[Dict[str, Any]]:
        """Parse relationships from model response."""
        relationships = []
        try:
            # Split response to get only the generated part
            generated_text = response.split("Relationships:")[1].strip()
            
            # Parse the response into structured format
            lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
            
            for line in lines:
                # Assuming format: "entity1 - relationship - entity2"
                parts = line.split('-')
                if len(parts) == 3:
                    relationships.append({
                        'source': parts[0].strip(),
                        'relationship': parts[1].strip(),
                        'target': parts[2].strip(),
                        'confidence': 0.8  # Could be calculated based on model output
                    })
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error parsing relationships: {str(e)}")
            return []

    def _calculate_confidence(self, entities: List[Dict[str, Any]], 
                            relationships: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the extraction results."""
        # This is a simplified confidence calculation
        # Could be enhanced based on model output scores
        
        if not entities:
            return 0.0
            
        # Basic confidence calculation
        entity_confidence = min(1.0, len(entities) / 10)  # Normalize by expected number
        relationship_confidence = min(1.0, len(relationships) / (len(entities) * 0.5))
        
        # Combined confidence score
        return (entity_confidence + relationship_confidence) / 2

# llm/prompts.py
class PromptTemplates:
    """Collection of prompt templates for different extraction tasks."""
    
    ENTITY_TYPES = [
        "PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT",
        "PRODUCT", "TECHNOLOGY", "CONCEPT"
    ]
    
    @staticmethod
    def get_entity_extraction_prompt() -> str:
        return """Analyze the following text and extract key entities. For each entity, specify its type from the following categories: PERSON, ORGANIZATION, LOCATION, DATE, EVENT, PRODUCT, TECHNOLOGY, CONCEPT.

Text: {text}

Format your response as:
Entity Type: Entity Text

Example:
PERSON: John Smith
ORGANIZATION: Acme Corp
DATE: January 2024

Extracted Entities:"""

    @staticmethod
    def get_relationship_extraction_prompt() -> str:
        return """Given the following text and identified entities, determine the relationships between them. Consider relationships such as:
- works_for
- located_in
- founded_by
- participated_in
- owns
- part_of

Text: {text}

Entities:
{entities}

Format your response as:
entity1 - relationship - entity2

Example:
John Smith - works_for - Acme Corp
Acme Corp - located_in - New York

Relationships:"""