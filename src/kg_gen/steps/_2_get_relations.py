import logging
from typing import List, Literal
import dspy
from pydantic import BaseModel
from ..utils.logging_config import setup_logger, log_operation
from ..utils.usage_tracker import usage_tracker
import mlflow

dspy.enable_logging()
logging.getLogger("dspy").setLevel(logging.DEBUG)

def extraction_sig(Relation: BaseModel, is_conversation: bool, context: str = "") -> dspy.Signature:
  if not is_conversation:
    
    class ExtractTextRelations(dspy.Signature):
      __doc__ = f"""Extract subject-predicate-object triples from the source text. 
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {context}"""
      
      source_text: str = dspy.InputField()
      entities: list[str] = dspy.InputField()
      relations: list[Relation] = dspy.OutputField(desc="List of subject-predicate-object tuples. Be thorough.")

    return ExtractTextRelations
  else:
    class ExtractConversationRelations(dspy.Signature):
      __doc__ = f"""Extract subject-predicate-object triples from the conversation, including:
      1. Relations between concepts discussed
      2. Relations between speakers and concepts (e.g. user asks about X)
      3. Relations between speakers (e.g. assistant responds to user)
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {context}"""
      
      source_text: str = dspy.InputField()
      entities: list[str] = dspy.InputField()
      relations: list[Relation] = dspy.OutputField(desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. Be thorough")
      
    return ExtractConversationRelations
        

def fallback_extraction_sig(entities, is_conversation, context: str = "") -> dspy.Signature:
  """This fallback extraction does not strictly type the subject and object strings."""
  
  entities_str = "\n- ".join(entities)

  class Relation(BaseModel):
    __doc__ = f"""Knowledge graph subject-predicate-object tuple. Subject and object entities must be one of: {entities_str}"""
    
    subject: str
    predicate: str
    object: str
    
  return Relation, extraction_sig(Relation, is_conversation, context)
  
@mlflow.trace
@log_operation("Relation Extraction")
def get_relations(dspy, input_data: str, entities: list[str], is_conversation: bool = False, context: str = "", log_level: int|str = "INFO") -> List[str]:
  """
  Extract relations between entities from input text or conversation.
  
  Args:
      dspy: DSPy runtime instance
      input_data: Text or conversation to process
      entities: List of entities to find relations between
      is_conversation: Whether input is a conversation format
      context: Additional context for relation extraction
      
  Returns:
      List of (subject, predicate, object) tuples
  """
  logger = setup_logger("kg_gen.relations", log_level=log_level)

  input_type = "conversation" if is_conversation else "text"
  logger.debug(f"Extracting relations from {input_type} using {len(entities)} entities")
  
  if not entities:
    logger.warning("No entities provided for relation extraction")
    return []

  class Relation(BaseModel):
    """Knowledge graph subject-predicate-object tuple."""
    subject: Literal[tuple(entities)]
    predicate: str
    object: Literal[tuple(entities)]
  
  ExtractRelations = extraction_sig(Relation, is_conversation, context)
  
  try:
    logger.debug("Attempting primary relation extraction")
    extract = dspy.Predict(ExtractRelations)
    result = extract(source_text=input_data, entities=entities)
    
    # Track usage with the global usage tracker
    usage_tracker.track_usage(result, logger=logger)
    
    relations = [(r.subject, r.predicate, r.object) for r in result.relations]
    logger.debug(f"Primary extraction successful: {len(relations)} relations found")
    return relations
  
  except Exception as e:
    logger.warning(f"Primary extraction failed ({str(e)}), using fallback approach")
    
    Relation, ExtractRelations = fallback_extraction_sig(entities, is_conversation, context)
    extract = dspy.Predict(ExtractRelations)
    result = extract(source_text=input_data, entities=entities)
    
    logger.debug(f"Fallback extraction produced {len(result.relations)} raw relations")
    
    class FixedRelations(dspy.Signature):
      """Fix the relations so that every subject and object of the relations are exact matches to an entity. Keep the predicate the same. The meaning of every relation should stay faithful to the reference text. If you cannot maintain the meaning of the original relation relative to the source text, then do not return it."""
      
      source_text: str = dspy.InputField()
      entities: list[str] = dspy.InputField()
      relations: list[Relation] = dspy.InputField()
      fixed_relations: list[Relation] = dspy.OutputField()
    
    fix = dspy.ChainOfThought(FixedRelations)
      
    fix_res = fix(source_text=input_data, entities=entities, relations=result.relations)
    
    # Track usage with the global usage tracker
    usage_tracker.track_usage(fix_res, logger=logger)

    good_relations = []
    for rel in fix_res.fixed_relations:
      if rel.subject in entities and rel.object in entities:
        good_relations.append(rel)
    
    logger.debug(f"After fixing and filtering: {len(good_relations)} valid relations")
    return [(r.subject, r.predicate, r.object) for r in good_relations]