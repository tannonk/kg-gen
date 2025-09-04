"""
Centralized logging configuration for kg-gen library.

This module provides standardized logging setup with structured formatting,
progress tracking, and different verbosity levels for knowledge graph operations.
"""

import logging
import sys
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager


class KGFormatter(logging.Formatter):
    """Custom formatter for kg-gen with structured output and color coding."""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors and structured information."""
        # Add color to levelname if outputting to terminal
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            colored_levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        else:
            colored_levelname = record.levelname
            
        # Create structured format
        formatted_time = self.formatTime(record, '%H:%M:%S')
        
        # Handle different record types
        if hasattr(record, 'operation'):
            # LLM operation log
            base_msg = f"[{formatted_time}] {colored_levelname:8} | {record.operation}"
            if hasattr(record, 'duration'):
                base_msg += f" ({record.duration:.2f}s)"
            if hasattr(record, 'details'):
                base_msg += f" | {record.details}"
            return base_msg
        elif hasattr(record, 'step'):
            # Pipeline step log  
            return f"[{formatted_time}] {colored_levelname:8} | Step: {record.step} | {record.getMessage()}"
        else:
            # Standard log
            return f"[{formatted_time}] {colored_levelname:8} | {record.getMessage()}"


def setup_logger(name: str = "kg_gen", log_level: int|str = "INFO") -> logging.Logger:
    """
    Set up a logger with standardized formatting for kg-gen operations.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
        
    # Set level
    if isinstance(log_level, int):
        logger.setLevel(log_level)
    else:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(KGFormatter())
    
    logger.addHandler(handler)
    logger.propagate = False  # Prevent duplicate logs from parent loggers
    
    return logger


def log_operation(operation_name: str, logger: Optional[logging.Logger] = None):
    """
    Decorator to log LLM operations with timing and details.
    
    Args:
        operation_name: Name of the operation being performed
        logger: Logger to use (if None, creates default)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_logger = logger or logging.getLogger("kg_gen")
            
            # Extract relevant details from arguments
            details = []
            if 'model' in kwargs:
                details.append(f"model={kwargs['model']}")
            if hasattr(args[0], 'model') and args[0].model:
                details.append(f"model={args[0].model}")
            
            # Log start of operation
            start_time = time.time()
            op_logger.info(
                f"Starting {operation_name}",
                extra={
                    'operation': operation_name,
                    'details': ' | '.join(details) if details else None
                }
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                duration = time.time() - start_time
                success_details = []
                if details:
                    success_details.extend(details)
                    
                # Add result details if available
                if hasattr(result, '__len__'):
                    success_details.append(f"result_size={len(result)}")
                elif isinstance(result, (list, set, tuple)):
                    success_details.append(f"count={len(result)}")
                    
                op_logger.info(
                    f"Completed {operation_name}",
                    extra={
                        'operation': operation_name,
                        'duration': duration,
                        'details': ' | '.join(success_details) if success_details else None
                    }
                )
                
                return result
                
            except Exception as e:
                # Log operation failure
                duration = time.time() - start_time
                op_logger.error(
                    f"Failed {operation_name}: {str(e)}",
                    extra={
                        'operation': operation_name,
                        'duration': duration,
                        'details': ' | '.join(details) if details else None
                    }
                )
                raise
                
        return wrapper
    return decorator


@contextmanager
def log_step(step_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager to log pipeline steps with timing.
    
    Args:
        step_name: Name of the pipeline step
        logger: Logger to use (if None, creates default)
    """
    step_logger = logger or logging.getLogger("kg_gen")
    
    start_time = time.time()
    step_logger.info(f"Starting {step_name}", extra={'step': step_name})
    
    try:
        yield step_logger
        duration = time.time() - start_time
        step_logger.info(
            f"Completed {step_name} in {duration:.2f}s",
            extra={'step': step_name}
        )
    except Exception as e:
        duration = time.time() - start_time
        step_logger.error(
            f"Failed {step_name} after {duration:.2f}s: {str(e)}",
            extra={'step': step_name}
        )
        raise


class ProgressTracker:
    """Simple progress tracker for operations with known total count."""
    
    def __init__(self, total: int, description: str = "Processing", 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of what's being processed
            logger: Logger to use for progress updates
        """
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logger or logging.getLogger("kg_gen")
        self.start_time = time.time()
        
    def update(self, increment: int = 1):
        """Update progress by increment amount."""
        self.current += increment
        
        # Calculate progress percentage
        progress = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        # Estimate remaining time
        if self.current > 0:
            estimated_total = elapsed * (self.total / self.current)
            remaining = estimated_total - elapsed
            eta_str = f" | ETA: {remaining:.1f}s" if remaining > 1 else ""
        else:
            eta_str = ""
            
        self.logger.info(
            f"{self.description}: {self.current}/{self.total} ({progress:.1f}%){eta_str}"
        )


def log_graph_stats(graph, operation: str = "Generated", logger: Optional[logging.Logger] = None):
    """
    Log statistics about a knowledge graph.
    
    Args:
        graph: Graph object with entities, relations, edges
        operation: Description of what operation produced this graph
        logger: Logger to use
    """
    stats_logger = logger or logging.getLogger("kg_gen")
    
    entities_count = len(graph.entities) if hasattr(graph, 'entities') else 0
    relations_count = len(graph.relations) if hasattr(graph, 'relations') else 0
    edges_count = len(graph.edges) if hasattr(graph, 'edges') else 0
    
    stats_logger.info(
        f"{operation} knowledge graph: {entities_count} entities, "
        f"{relations_count} relations, {edges_count} unique edge types"
    )
    
    # Log clustering info if available
    if hasattr(graph, 'entity_clusters') and graph.entity_clusters:
        clustered_entities = sum(len(cluster) for cluster in graph.entity_clusters.values())
        stats_logger.debug(
            f"Entity clustering: {len(graph.entity_clusters)} clusters covering "
            f"{clustered_entities} entities"
        )
        
    if hasattr(graph, 'edge_clusters') and graph.edge_clusters:
        clustered_edges = sum(len(cluster) for cluster in graph.edge_clusters.values()) 
        stats_logger.debug(
            f"Edge clustering: {len(graph.edge_clusters)} clusters covering "
            f"{clustered_edges} edges"
        )