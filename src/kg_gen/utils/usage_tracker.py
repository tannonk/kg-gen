"""
Usage tracking utilities for KG-Gen.

This module provides a thread-safe singleton UsageTracker class that accumulates
DSPy language model usage statistics across the entire program lifecycle.
"""

import threading
import json
import csv
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# pricing
MODEL_PRICING = {
    # https://platform.openai.com/docs/pricing
    "openai/gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-5-nano-2025-08-07": {
        "input": 0.05,
        "output": 0.40,
    },  # https://platform.openai.com/docs/models/gpt-5-nano
    "gemini/gemini-2.5-pro": {
        "input": 0.00,
        "output": 0.00,
    },  # https://ai.google.dev/gemini-api/docs/pricing
    "gemini/gemini-2.5-flash": {
        "input": 0.00,
        "output": 0.00,
    },  # https://ai.google.dev/gemini-api/docs/pricing
}


@dataclass
class ModelStats:
    """Statistics for a specific language model."""

    model_name: str
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    first_call_time: Optional[datetime] = None
    last_call_time: Optional[datetime] = None
    errors: int = 0

    def add_usage(self, usage_data: Dict[str, Any]) -> None:
        """Add usage statistics from a DSPy result."""
        now = datetime.now()

        if self.first_call_time is None:
            self.first_call_time = now
        self.last_call_time = now

        self.total_calls += 1

        # Handle different possible usage data formats
        if isinstance(usage_data, dict):
            # Extract token counts - prioritize main fields
            prompt_tokens = (
                usage_data.get("prompt_tokens", 0)
                or usage_data.get("input_tokens", 0)
                or 0
            )
            completion_tokens = (
                usage_data.get("completion_tokens", 0)
                or usage_data.get("output_tokens", 0)
                or 0
            )
            total_tokens = usage_data.get("total_tokens", 0) or (
                prompt_tokens + completion_tokens
            )
            cost = usage_data.get("cost", 0.0) or 0.0

            # Ensure we have valid numeric values
            prompt_tokens = int(prompt_tokens) if prompt_tokens else 0
            completion_tokens = int(completion_tokens) if completion_tokens else 0
            total_tokens = int(total_tokens) if total_tokens else 0
            cost = float(cost) if cost else 0.0

            # If no cost provided, calculate using model pricing
            if cost == 0.0 and self.model_name in MODEL_PRICING:
                pricing = MODEL_PRICING[self.model_name]
                input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
                output_cost = (completion_tokens / 1_000_000) * pricing["output"]
                cost = input_cost + output_cost

            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.total_cost += cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        if self.first_call_time:
            data["first_call_time"] = self.first_call_time.isoformat()
        if self.last_call_time:
            data["last_call_time"] = self.last_call_time.isoformat()
        return data


class UsageTracker:
    """
    Thread-safe singleton class for tracking DSPy language model usage across the program.

    Features:
    - Singleton pattern ensures one instance across all components
    - Thread-safe for concurrent operations
    - Per-model statistics tracking
    - Aggregate statistics calculation
    - Export to multiple formats (JSON, CSV, human-readable)
    - Integration with logging framework
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(UsageTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the usage tracker."""
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._stats_lock = threading.Lock()
        self._model_stats: Dict[str, ModelStats] = {}
        self._session_start = datetime.now()
        self._logger = logging.getLogger(__name__)

    def track_usage(
        self,
        result,
        step: Optional[str] = "UnknownStep",
        model_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Track usage from a DSPy result object.

        Args:
            result: DSPy result object with get_lm_usage() method
            model_name: Optional model name override
            logger: Optional logger for backward compatibility
        """
        try:
            usage_data = (
                result.get_lm_usage() if hasattr(result, "get_lm_usage") else {}
            )

            # Log usage for backward compatibility (if logger provided)
            if logger and usage_data:
                logger.info(f"LM Usage ({step}): {usage_data}")

            if not usage_data:
                return

            # Handle the nested structure: {'model_name': {'completion_tokens': X, 'prompt_tokens': Y, ...}}
            if isinstance(usage_data, dict):
                for detected_model_name, model_usage in usage_data.items():
                    # Use provided model_name if available, otherwise use detected model name
                    final_model_name = model_name or detected_model_name

                    if isinstance(model_usage, dict):
                        self._add_model_usage(final_model_name, model_usage)
                    else:
                        self._logger.warning(
                            f"Unexpected usage data format for model {detected_model_name}: {model_usage}"
                        )
            else:
                self._logger.warning(f"Unexpected usage data format: {usage_data}")

        except Exception as e:
            self._logger.warning(f"Failed to track usage: {e}")

    def _add_model_usage(self, model_name: str, usage_data: Dict[str, Any]) -> None:
        """Thread-safe method to add usage for a model."""
        with self._stats_lock:
            if model_name not in self._model_stats:
                self._model_stats[model_name] = ModelStats(model_name=model_name)

            self._model_stats[model_name].add_usage(usage_data)

    def get_model_stats(self, model_name: str) -> Optional[ModelStats]:
        """Get statistics for a specific model."""
        with self._stats_lock:
            return self._model_stats.get(model_name)

    def get_all_stats(self) -> Dict[str, ModelStats]:
        """Get statistics for all tracked models."""
        with self._stats_lock:
            return self._model_stats.copy()

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all models."""
        with self._stats_lock:
            if not self._model_stats:
                return {}

            total_calls = sum(stats.total_calls for stats in self._model_stats.values())
            total_tokens = sum(
                stats.total_tokens for stats in self._model_stats.values()
            )
            total_cost = sum(stats.total_cost for stats in self._model_stats.values())
            total_errors = sum(stats.errors for stats in self._model_stats.values())

            # Calculate time span
            all_first_times = [
                stats.first_call_time
                for stats in self._model_stats.values()
                if stats.first_call_time
            ]
            all_last_times = [
                stats.last_call_time
                for stats in self._model_stats.values()
                if stats.last_call_time
            ]

            first_call = min(all_first_times) if all_first_times else None
            last_call = max(all_last_times) if all_last_times else None

            return {
                "session_start": self._session_start.isoformat(),
                "total_models": len(self._model_stats),
                "total_calls": total_calls,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "total_errors": total_errors,
                "first_call": first_call.isoformat() if first_call else None,
                "last_call": last_call.isoformat() if last_call else None,
                "duration_seconds": (last_call - first_call).total_seconds()
                if first_call and last_call
                else 0,
                "models": list(self._model_stats.keys()),
            }

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._stats_lock:
            self._model_stats.clear()
            self._session_start = datetime.now()

    def export_json(self, include_aggregate: bool = True) -> str:
        """Export statistics as JSON string."""
        data = {}

        # Add per-model stats
        data["models"] = {
            name: stats.to_dict() for name, stats in self.get_all_stats().items()
        }

        # Add aggregate stats
        if include_aggregate:
            data["aggregate"] = self.get_aggregate_stats()

        return json.dumps(data, indent=2, default=str)

    def export_csv(self, file_path: str) -> None:
        """Export statistics to CSV file."""
        stats = self.get_all_stats()
        if not stats:
            return

        with open(file_path, "w", newline="") as csvfile:
            fieldnames = [
                "model_name",
                "total_calls",
                "total_prompt_tokens",
                "total_completion_tokens",
                "total_tokens",
                "total_cost",
                "first_call_time",
                "last_call_time",
                "errors",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for stats in self.get_all_stats().values():
                row = stats.to_dict()
                writer.writerow(row)

    def get_summary(self) -> str:
        """Get human-readable summary of usage statistics."""
        aggregate = self.get_aggregate_stats()
        model_stats = self.get_all_stats()

        if not aggregate:
            return "No usage data tracked."

        lines = [
            "=== KG-Gen Usage Summary ===",
            f"Session Duration: {aggregate.get('duration_seconds', 0):.1f} seconds",
            f"Total Models Used: {aggregate['total_models']}",
            f"Total API Calls: {aggregate['total_calls']}",
            f"Total Tokens: {aggregate['total_tokens']:,}",
            f"Total Cost: ${aggregate['total_cost']:.4f}",
            "",
        ]

        if model_stats:
            lines.append("Per-Model Breakdown:")
            for model_name, stats in model_stats.items():
                # Determine if cost is estimated or provided
                has_pricing = model_name in MODEL_PRICING
                cost_note = " (estimated)" if has_pricing else ""

                lines.extend(
                    [
                        f"  {model_name}:",
                        f"    Calls: {stats.total_calls}",
                        f"    Tokens: {stats.total_tokens:,} (prompt: {stats.total_prompt_tokens:,}, completion: {stats.total_completion_tokens:,})",
                        f"    Cost: ${stats.total_cost:.4f}{cost_note}",
                        "",
                    ]
                )

        return "\n".join(lines)

    def log_summary(self, logger: Optional[logging.Logger] = None) -> None:
        """Log usage summary."""
        target_logger = logger or self._logger
        target_logger.info(f"\n{self.get_summary()}")


def get_model_pricing_info() -> Dict[str, Dict[str, float]]:
    """
    Get the current model pricing information.

    Returns:
        Dictionary mapping model names to their pricing info
    """
    return MODEL_PRICING.copy()


def add_model_pricing(model_name: str, input_price: float, output_price: float) -> None:
    """
    Add or update pricing for a model.

    Args:
        model_name: Name of the model
        input_price: Price per million input tokens (USD)
        output_price: Price per million output tokens (USD)
    """
    MODEL_PRICING[model_name] = {"input": input_price, "output": output_price}


# Global singleton instance
usage_tracker = UsageTracker()
