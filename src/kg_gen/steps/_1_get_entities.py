import logging
from typing import List
import dspy
from ..utils.logging_config import setup_logger, log_operation
from ..utils.usage_tracker import usage_tracker
import mlflow

logging.getLogger("dspy").setLevel(logging.DEBUG)

class TextEntities(dspy.Signature):
  """Extract key entities from the source text. Extracted entities are subjects or objects.
  This is for an extraction task, please be THOROUGH and accurate to the reference text."""

  source_text: str = dspy.InputField()
  entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

class ConversationEntities(dspy.Signature):
  """Extract key entities from the conversation Extracted entities are subjects or objects.
  Consider both explicit entities and participants in the conversation.
  This is for an extraction task, please be THOROUGH and accurate."""
  
  source_text: str = dspy.InputField()
  entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")
  
@mlflow.trace
@log_operation("Entity Extraction")
def get_entities(dspy, input_data: str, is_conversation: bool = False, log_level: int|str = "INFO") -> List[str]:
  """
  Extract entities from input text or conversation.
  
  Args:
      dspy: DSPy runtime instance
      input_data: Text or conversation to process
      is_conversation: Whether input is a conversation format
      
  Returns:
      List of extracted entity strings
  """
  logger = setup_logger("kg_gen.entities", log_level=log_level)
  
  input_type = "conversation" if is_conversation else "text"
  logger.debug(f"Extracting entities from {input_type} ({len(input_data)} chars)")
  
  if is_conversation:
    extract = dspy.Predict(ConversationEntities)
  else:
    extract = dspy.Predict(TextEntities)
  
  result = extract(source_text=input_data)
  
  # Track usage with the global usage tracker
  usage_tracker.track_usage(result, step="ExtractEntities", logger=logger)
  
  entities = result.entities if result.entities else []
  logger.debug(f"Extracted {len(entities)} entities: {entities[:5]}{'...' if len(entities) > 5 else ''}")
  
  return entities

