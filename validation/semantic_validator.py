# validation/semantic_validator.py
class SemanticValidator:
    def __init__(self, config):
        self.ontology = OntologyLoader.load(config.ontology_path)
        self.rules = RuleEngine(config.rules_path)
        
    def validate_triple(self, subject, predicate, object):
        validations = [
            self.ontology.validate_types(subject.type, object.type),
            self.rules.validate_relation(predicate),
            self.ontology.validate_cardinality(subject, predicate, object)
        ]
        return all(validations)