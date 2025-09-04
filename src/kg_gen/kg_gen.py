from typing import Union, List, Dict, Optional
from openai import OpenAI

from .steps._1_get_entities import get_entities
from .steps._2_get_relations import get_relations
from .steps._3_cluster_graph import cluster_graph
from .utils.chunk_text import chunk_text
from .utils.logging_config import setup_logger, log_step, log_graph_stats, ProgressTracker
from .utils.usage_tracker import usage_tracker, get_model_pricing_info, add_model_pricing
from .models import Graph
import dspy
import mlflow
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor

logging.getLogger("dspy").setLevel(logging.DEBUG)

class KGGen:
  def __init__(
    self,
    model: str = "openai/gpt-4o",
    temperature: float = 0.0,
    api_key: str = None,
    api_base: str = None,
    max_tokens: int = 4000,
    log_level: int|str = "INFO",
    force: bool = False
  ):
    """Initialize KGGen with optional model configuration and logging

    Args:
        model: Name of model to use (e.g. 'gpt-4')
        temperature: Temperature for model sampling
        api_key: API key for model access
        api_base: Specify the base URL endpoint for making API calls to a language model service
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        force: If true, cache will be bypassed and fresh API calls will be made
    """
    # Set up logging
    self.logger = setup_logger("kg_gen.main", log_level)
    
    self.dspy = dspy
    self.model = model
    self.temperature = temperature
    self.api_key = api_key
    self.api_base = api_base
    self.max_tokens = max_tokens
    self.force = force

    # Log initialization
    self.logger.info(f"Initializing KGGen with model: {model}")
    if api_base:
      self.logger.debug(f"Using custom API base: {api_base}")

    self.init_model(model=model, api_key=api_key, api_base=api_base, temperature=temperature, max_tokens=max_tokens)
    self.usage = self.dspy.settings.get("usage", {})

  def init_model(
    self,
    model: str = None,
    api_key: str = None,
    api_base: str = None,
    temperature: float = None,
    max_tokens: int = None,
  ):
    """Initialize or reinitialize the model with new parameters

    Args:
        model: Name of model to use (e.g. 'gpt-4')
        temperature: Temperature for model sampling
        api_key: API key for model access
        api_base: API base for model access
    """
    # Update instance variables if new values provided
    if model is not None:
      self.model = model
    if temperature is not None:
      self.temperature = temperature
    if api_key is not None:
      self.api_key = api_key
    if api_base is not None:
      self.api_base = api_base
    if max_tokens is not None:
      self.max_tokens = max_tokens

    # Log model configuration
    self.logger.debug(f"Configuring model: {self.model} (temp={self.temperature})")
    
    # Initialize dspy LM with current settings
    if self.api_key:
      self.lm = dspy.LM(model=self.model, api_key=self.api_key, temperature=self.temperature, api_base=self.api_base, max_tokens=self.max_tokens)
    else:
      self.lm = dspy.LM(model=self.model, temperature=self.temperature, api_base=self.api_base, max_tokens=self.max_tokens)

    self.dspy.configure(lm=self.lm, cache=self.force)
    
    self.dspy.settings.configure(track_usage=True)

    # Disable caching to ensure fresh API calls (necessary for iterative clustering steps)
    self.dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
    )

    # if self.force:
    #   # disable cache usage to ensure fresh API calls
    #   self.dspy.configure_cache(
    #       enable_disk_cache=False,
    #       enable_memory_cache=False,
    #   )
    # else:
    #   # enable cache usage for improved performance
    #   self.dspy.configure_cache(
    #       enable_disk_cache=True,
    #       enable_memory_cache=True,
    #       disk_size_limit_bytes=1_000_000_000, # 1GB
    #       memory_max_entries=1_000_000,
    #   )
    self.dspy.enable_logging()
    
    self.logger.debug("DSPy LM configuration completed")

  @mlflow.trace
  def generate(
    self,
    input_data: Union[str, List[Dict]],
    model: str = None,
    api_key: str = None,
    api_base: str = None,
    context: str = "",
    # example_relations: Optional[Union[
    #   List[Tuple[str, str, str]],
    #   List[Tuple[Tuple[str, str], str, Tuple[str, str]]]
    # ]] = None,
    chunk_size: Optional[int] = None,
    cluster: bool = False,
    temperature: float = None,
    # node_labels: Optional[List[str]] = None,
    # edge_labels: Optional[List[str]] = None,
    # ontology: Optional[List[Tuple[str, str, str]]] = None,
    output_folder: Optional[str] = None,
    max_tokens: int = None,
  ) -> Graph:
    """Generate a knowledge graph from input text or messages.

    Args:
        input_data: Text string or list of message dicts
        model: Name of OpenAI model to use
        api_key (str): OpenAI API key for making model calls
        chunk_size: Max size of text chunks in characters to process
        context: Description of data context
        example_relations: Example relationship tuples
        node_labels: Valid node label strings
        edge_labels: Valid edge label strings
        ontology: Valid node-edge-node structure tuples
        output_folder: Path to save partial progress

    Returns:
        Generated knowledge graph
    """
    
    with log_step("Knowledge Graph Generation", self.logger):
      # Process input data
      is_conversation = isinstance(input_data, list)
      input_type = "conversation" if is_conversation else "text"
      
      if is_conversation:
        # Extract text from messages
        text_content = []
        for message in input_data:
          if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Messages must be dicts with 'role' and 'content' keys")
          if message['role'] in ['user', 'assistant']:
            text_content.append(f"{message['role']}: {message['content']}")

        # Join with newlines to preserve message boundaries
        processed_input = "\n".join(text_content)
        self.logger.info(f"Processing conversation with {len(input_data)} messages")
      else:
        processed_input = input_data
        self.logger.info(f"Processing text input ({len(processed_input)} characters)")

      # Reinitialize dspy with new parameters if any are provided
      if any([model, temperature, api_key, api_base, max_tokens]):
        with log_step("Model Reconfiguration", self.logger):
          self.init_model(
            model=model or self.model,
            temperature=temperature or self.temperature,
            api_key=api_key or self.api_key,
            api_base=api_base or self.api_base,
            max_tokens=max_tokens or self.max_tokens,
          )

      # Knowledge extraction process
      if not chunk_size:
        # Single-pass processing
        with log_step("Entity Extraction", self.logger):
          entities = get_entities(self.dspy, processed_input, is_conversation=is_conversation, log_level=self.logger.level)
          self.logger.info(f"Extracted {len(entities)} entities")
          
        with log_step("Relation Extraction", self.logger):
          relations = get_relations(self.dspy, processed_input, entities, is_conversation=is_conversation, log_level=self.logger.level)
          self.logger.info(f"Extracted {len(relations)} relations")
      else:
        # Chunked processing
        with log_step("Text Chunking", self.logger):
          chunks = chunk_text(processed_input, chunk_size)
          self.logger.info(f"Split input into {len(chunks)} chunks (max size: {chunk_size})")
          
        entities = set()
        relations = set()

        def process_chunk(chunk_idx_and_chunk):
          chunk_idx, chunk = chunk_idx_and_chunk
          self.logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} chars)")
          chunk_entities = get_entities(self.dspy, chunk, is_conversation=is_conversation, log_level=self.logger.level)
          chunk_relations = get_relations(self.dspy, chunk, chunk_entities, is_conversation=is_conversation, log_level=self.logger.level)
          return chunk_entities, chunk_relations

        # Process chunks in parallel using ThreadPoolExecutor
        with log_step("Parallel Chunk Processing", self.logger):
          progress = ProgressTracker(len(chunks), "Processing chunks", self.logger)
          
          with ThreadPoolExecutor() as executor:
            indexed_chunks = list(enumerate(chunks))
            results = list(executor.map(process_chunk, indexed_chunks))
            
          # Combine results
          for i, (chunk_entities, chunk_relations) in enumerate(results):
            entities.update(chunk_entities)
            relations.update(chunk_relations)
            progress.update()
            
          self.logger.info(f"Combined results: {len(entities)} unique entities, {len(relations)} unique relations")

      # Create initial graph
      graph = Graph(
        entities = entities,
        relations = relations,
        edges = {relation[1] for relation in relations}
      )
      
      log_graph_stats(graph, "Initial", self.logger)

      # Optional clustering step
      if cluster:
        with log_step("Graph Clustering", self.logger):
          graph = self.cluster(graph, context, log_level=self.logger.level)
          log_graph_stats(graph, "Clustered", self.logger)

      # Optional output saving
      if output_folder:
        with log_step("Saving Output", self.logger):
          os.makedirs(output_folder, exist_ok=True)
          output_path = os.path.join(output_folder, 'graph.json')

          graph_dict = {
            'entities': list(entities),
            'relations': list(relations),
            'edges': list(graph.edges)
          }

          with open(output_path, 'w') as f:
            json.dump(graph_dict, f, indent=2)
          
          self.logger.info(f"Saved graph to {output_path}")

      return graph

  def cluster(
    self,
    graph: Graph,
    context: str = "",
    entity_clustering_context: str = "",
    relation_clustering_context: str = "",
    skip_entity_clustering: bool = False,
    skip_relation_clustering: bool = False,
    model: str = None,
    temperature: float = None,
    api_key: str = None,
    api_base: str = None,
    max_tokens: int = None,
    log_level: int|str = "INFO"
  ) -> Graph:
    """
    Cluster entities and relations in a knowledge graph for deduplication.
    
    Args:
        graph: Input graph to cluster
        context: Additional context for clustering decisions
        model: Optional model override
        temperature: Optional temperature override
        api_key: Optional API key override
        api_base: Optional API base override
        init_kwargs: Optional model initialization kwargs
        
    Returns:
        Graph with clustered entities and relations
    """
    self.logger.info(f"Starting graph clustering...")

    # Reinitialize dspy with new parameters if any are provided
    if any([model, temperature, api_key, api_base, max_tokens]):
      with log_step("Model Reconfiguration for Clustering", self.logger):
        self.init_model(
          model=model or self.model,
          temperature=temperature or self.temperature,
          api_key=api_key or self.api_key,
          api_base=api_base or self.api_base,
          max_tokens=max_tokens or self.max_tokens,
        )

    return cluster_graph(self.dspy, graph=graph, context=context, entity_cluster_context=entity_clustering_context,
                         relation_cluster_context=relation_clustering_context,
                         skip_entity_clustering=skip_entity_clustering,
                         skip_relation_clustering=skip_relation_clustering,
                         log_level=log_level)

  def aggregate(self, graphs: list[Graph]) -> Graph:
    """
    Aggregate multiple knowledge graphs into a single combined graph.
    
    Args:
        graphs: List of Graph objects to combine
        
    Returns:
        Single aggregated Graph object
    """
    with log_step("Graph Aggregation", self.logger):
      self.logger.info(f"Aggregating {len(graphs)} graphs")
      
      # Initialize empty sets for combined graph
      all_entities = set()
      all_relations = set()
      all_edges = set()

      # Combine all graphs
      for i, graph in enumerate(graphs):
        entities_before = len(all_entities)
        relations_before = len(all_relations) 
        edges_before = len(all_edges)
        
        all_entities.update(graph.entities)
        all_relations.update(graph.relations)
        all_edges.update(graph.edges)
        
        entities_added = len(all_entities) - entities_before
        relations_added = len(all_relations) - relations_before
        edges_added = len(all_edges) - edges_before
        
        self.logger.debug(
          f"Graph {i+1}: +{entities_added} entities, "
          f"+{relations_added} relations, +{edges_added} edges"
        )

      # Create and return aggregated graph
      aggregated_graph = Graph(
        entities=all_entities,
        relations=all_relations,
        edges=all_edges
      )
      
      log_graph_stats(aggregated_graph, "Aggregated", self.logger)
      return aggregated_graph

  def get_usage_stats(self) -> Dict[str, any]:
    """
    Get comprehensive usage statistics for all models used during this session.
    
    Returns:
        Dictionary containing per-model and aggregate usage statistics
    """
    return {
      'models': {name: stats.to_dict() for name, stats in usage_tracker.get_all_stats().items()},
      'aggregate': usage_tracker.get_aggregate_stats()
    }
  
  def get_usage_summary(self) -> str:
    """
    Get a human-readable summary of usage statistics.
    
    Returns:
        Formatted string with usage summary
    """
    return usage_tracker.get_summary()
  
  def log_usage_summary(self) -> None:
    """Log usage summary to the configured logger."""
    usage_tracker.log_summary(self.logger)
  
  def export_usage_json(self, include_aggregate: bool = True) -> str:
    """
    Export usage statistics as JSON string.
    
    Args:
        include_aggregate: Whether to include aggregate statistics
        
    Returns:
        JSON string containing usage statistics
    """
    return usage_tracker.export_json(include_aggregate=include_aggregate)
  
  def export_usage_csv(self, file_path: str) -> None:
    """
    Export usage statistics to CSV file.
    
    Args:
        file_path: Path to save the CSV file
    """
    usage_tracker.export_csv(file_path)
  
  def reset_usage_tracking(self) -> None:
    """Reset all usage tracking data."""
    usage_tracker.reset()
    self.logger.info("Usage tracking data has been reset")
  
  def get_model_pricing_info(self) -> Dict[str, Dict[str, float]]:
    """
    Get current model pricing information.
    
    Returns:
        Dictionary mapping model names to pricing info (input/output per million tokens)
    """
    return get_model_pricing_info()
  
  def add_model_pricing(self, model_name: str, input_price: float, output_price: float) -> None:
    """
    Add or update pricing for a model.
    
    Args:
        model_name: Name of the model 
        input_price: Price per million input tokens (USD)
        output_price: Price per million output tokens (USD)
    """
    add_model_pricing(model_name, input_price, output_price)
    self.logger.info(f"Updated pricing for {model_name}: input=${input_price}/M, output=${output_price}/M")
