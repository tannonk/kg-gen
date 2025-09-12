"""
Unit tests for the UsageTracker functionality.

This module tests the singleton usage tracking system, including:
- Thread safety
- Data accumulation
- Statistics calculation
- Export functionality
- Integration with DSPy
"""

import pytest
import json
import os
import tempfile
import threading
import time
from unittest.mock import Mock
from datetime import datetime

from kg_gen.utils.usage_tracker import UsageTracker, ModelStats, usage_tracker
from kg_gen import KGGen


class TestUsageTracker:
    """Test the core UsageTracker functionality."""

    def setup_method(self):
        """Reset usage tracker before each test."""
        usage_tracker.reset()

    def test_singleton_pattern(self):
        """Test that UsageTracker follows singleton pattern."""
        tracker1 = UsageTracker()
        tracker2 = UsageTracker()

        assert tracker1 is tracker2
        assert tracker1 is usage_tracker

    def test_model_stats_initialization(self):
        """Test ModelStats initialization and basic operations."""
        stats = ModelStats(model_name="test-model")

        assert stats.model_name == "test-model"
        assert stats.total_calls == 0
        assert stats.total_tokens == 0
        assert stats.total_cost == 0.0
        assert stats.first_call_time is None
        assert stats.last_call_time is None

    def test_model_stats_add_usage(self):
        """Test adding usage data to ModelStats."""
        stats = ModelStats(model_name="test-model")

        # Test standard usage format
        usage_data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.01,
        }

        stats.add_usage(usage_data)

        assert stats.total_calls == 1
        assert stats.total_prompt_tokens == 100
        assert stats.total_completion_tokens == 50
        assert stats.total_tokens == 150
        assert stats.total_cost == 0.01
        assert stats.first_call_time is not None
        assert stats.last_call_time is not None

    def test_model_stats_alternative_token_names(self):
        """Test ModelStats handles alternative token field names."""
        stats = ModelStats(model_name="test-model")

        # Test alternative naming convention
        usage_data = {"input_tokens": 80, "output_tokens": 40, "cost": 0.008}

        stats.add_usage(usage_data)

        assert stats.total_prompt_tokens == 80
        assert stats.total_completion_tokens == 40
        assert stats.total_tokens == 120  # Should sum input + output
        assert stats.total_cost == 0.008

    def test_model_stats_multiple_calls(self):
        """Test accumulating stats across multiple calls."""
        stats = ModelStats(model_name="test-model")

        # Add first usage
        stats.add_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cost": 0.01,
            }
        )

        # Add second usage
        stats.add_usage(
            {
                "prompt_tokens": 80,
                "completion_tokens": 40,
                "total_tokens": 120,
                "cost": 0.008,
            }
        )

        assert stats.total_calls == 2
        assert stats.total_prompt_tokens == 180
        assert stats.total_completion_tokens == 90
        assert stats.total_tokens == 270
        assert stats.total_cost == 0.018

    def test_track_usage_with_mock_result(self):
        """Test tracking usage from a mock DSPy result."""
        tracker = UsageTracker()

        # Mock DSPy result with nested structure like actual DSPy
        mock_result = Mock()
        mock_result.get_lm_usage.return_value = {
            "openai/gpt-4.1-nano-2025-04-14": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cost": 0.01,
            }
        }

        tracker.track_usage(mock_result, model_name="test-model")

        stats = tracker.get_model_stats("test-model")
        assert stats is not None
        assert stats.total_calls == 1
        assert stats.total_tokens == 150
        assert stats.total_cost == 0.01

    def test_track_usage_without_model_name(self):
        """Test tracking usage when model name is extracted from usage data."""
        tracker = UsageTracker()

        mock_result = Mock()
        mock_result.get_lm_usage.return_value = {
            "openai/extracted-model": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cost": 0.01,
            }
        }

        tracker.track_usage(mock_result)  # No model_name provided

        stats = tracker.get_model_stats("openai/extracted-model")
        assert stats is not None
        assert stats.total_calls == 1

    def test_track_usage_empty_data(self):
        """Test tracking usage with empty data."""
        tracker = UsageTracker()

        mock_result = Mock()
        mock_result.get_lm_usage.return_value = {}

        tracker.track_usage(mock_result, model_name="test-model")

        # Should not create stats for empty data
        stats = tracker.get_model_stats("test-model")
        assert stats is None

    def test_track_usage_with_logger(self):
        """Test tracking usage with logger integration."""
        tracker = UsageTracker()
        mock_logger = Mock()

        mock_result = Mock()
        mock_result.get_lm_usage.return_value = {
            "openai/test-model": {"prompt_tokens": 100, "completion_tokens": 50}
        }

        tracker.track_usage(mock_result, model_name="test-model", logger=mock_logger)

        # Should log usage for backward compatibility
        mock_logger.info.assert_called_once()

    def test_get_aggregate_stats(self):
        """Test aggregate statistics calculation."""
        tracker = UsageTracker()

        # Add usage for multiple models
        mock_result1 = Mock()
        mock_result1.get_lm_usage.return_value = {
            "openai/model1": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost": 0.01,
            }
        }

        mock_result2 = Mock()
        mock_result2.get_lm_usage.return_value = {
            "openai/model2": {
                "prompt_tokens": 80,
                "completion_tokens": 40,
                "cost": 0.008,
            }
        }

        tracker.track_usage(mock_result1)
        tracker.track_usage(mock_result2)

        aggregate = tracker.get_aggregate_stats()

        assert aggregate["total_models"] == 2
        assert aggregate["total_calls"] == 2
        assert aggregate["total_tokens"] == 270  # 150 + 120
        assert aggregate["total_cost"] == 0.018  # 0.01 + 0.008
        assert set(aggregate["models"]) == {"openai/model1", "openai/model2"}

    def test_reset(self):
        """Test resetting tracker data."""
        tracker = UsageTracker()

        # Add some data
        mock_result = Mock()
        mock_result.get_lm_usage.return_value = {
            "openai/test-model": {"prompt_tokens": 100, "completion_tokens": 50}
        }

        tracker.track_usage(mock_result)

        assert len(tracker.get_all_stats()) == 1

        # Reset and verify
        tracker.reset()

        assert len(tracker.get_all_stats()) == 0
        aggregate = tracker.get_aggregate_stats()
        assert aggregate == {}

    def test_export_json(self):
        """Test JSON export functionality."""
        tracker = UsageTracker()

        # Add test data
        mock_result = Mock()
        mock_result.get_lm_usage.return_value = {
            "openai/test-model": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost": 0.01,
            }
        }

        tracker.track_usage(mock_result)

        # Test JSON export
        json_data = tracker.export_json()
        parsed = json.loads(json_data)

        assert "models" in parsed
        assert "aggregate" in parsed
        assert "openai/test-model" in parsed["models"]
        assert parsed["models"]["openai/test-model"]["total_calls"] == 1
        assert parsed["aggregate"]["total_tokens"] == 150

    def test_export_json_without_aggregate(self):
        """Test JSON export without aggregate stats."""
        tracker = UsageTracker()

        mock_result = Mock()
        mock_result.get_lm_usage.return_value = {
            "openai/test-model": {"prompt_tokens": 100, "completion_tokens": 50}
        }

        tracker.track_usage(mock_result)

        json_data = tracker.export_json(include_aggregate=False)
        parsed = json.loads(json_data)

        assert "models" in parsed
        assert "aggregate" not in parsed

    def test_export_csv(self):
        """Test CSV export functionality."""
        tracker = UsageTracker()

        # Add test data
        mock_result = Mock()
        mock_result.get_lm_usage.return_value = {
            "openai/test-model": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost": 0.01,
            }
        }

        tracker.track_usage(mock_result)

        # Test CSV export
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            csv_path = f.name

        try:
            tracker.export_csv(csv_path)

            # Verify CSV content
            with open(csv_path, "r") as f:
                content = f.read()
                assert "model_name" in content  # Header
                assert "openai/test-model" in content  # Data
                assert "100" in content  # Token count
        finally:
            os.unlink(csv_path)

    def test_get_summary(self):
        """Test human-readable summary generation."""
        tracker = UsageTracker()

        # Add test data
        mock_result = Mock()
        mock_result.get_lm_usage.return_value = {
            "openai/test-model": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost": 0.01,
            }
        }

        tracker.track_usage(mock_result)

        summary = tracker.get_summary()

        assert "KG-Gen Usage Summary" in summary
        assert "Total Models Used: 1" in summary
        assert "Total API Calls: 1" in summary
        assert "openai/test-model" in summary
        assert "150" in summary  # Total tokens

    def test_get_summary_empty(self):
        """Test summary with no data."""
        tracker = UsageTracker()

        summary = tracker.get_summary()
        assert "No usage data tracked" in summary

    def test_thread_safety(self):
        """Test thread safety of the usage tracker."""
        tracker = UsageTracker()
        results = []
        errors = []

        def track_usage_thread(thread_id):
            try:
                for i in range(10):
                    mock_result = Mock()
                    mock_result.get_lm_usage.return_value = {
                        f"openai/model-{thread_id}": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "cost": 0.001,
                        }
                    }

                    tracker.track_usage(mock_result)
                    time.sleep(0.001)  # Small delay to encourage race conditions

                results.append(thread_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=track_usage_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors and all data tracked correctly
        assert len(errors) == 0
        assert len(results) == 5

        # Check that each model has correct stats
        all_stats = tracker.get_all_stats()
        assert len(all_stats) == 5

        for i in range(5):
            model_stats = tracker.get_model_stats(f"openai/model-{i}")
            assert model_stats is not None
            assert model_stats.total_calls == 10


class TestKGGenUsageIntegration:
    """Test UsageTracker integration with KGGen class."""

    def setup_method(self):
        """Reset usage tracker before each test."""
        usage_tracker.reset()

    def test_kg_gen_usage_methods(self):
        """Test KGGen usage tracking methods."""
        # Create KGGen instance with dummy configuration
        kg_gen = KGGen(model="test/model", temperature=0.0, api_key="dummy-key")

        # Mock some usage data by directly adding to tracker
        mock_result = Mock()
        mock_result.get_lm_usage.return_value = {
            "test/model": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
        }

        usage_tracker.track_usage(mock_result)

        # Test KGGen methods
        stats = kg_gen.get_usage_stats()
        assert "models" in stats
        assert "aggregate" in stats
        assert "test/model" in stats["models"]

        summary = kg_gen.get_usage_summary()
        assert "KG-Gen Usage Summary" in summary
        assert "test/model" in summary

        # Test JSON export
        json_export = kg_gen.export_usage_json()
        parsed = json.loads(json_export)
        assert "models" in parsed

        # Test CSV export
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            csv_path = f.name

        try:
            kg_gen.export_usage_csv(csv_path)
            assert os.path.exists(csv_path)
        finally:
            os.unlink(csv_path)

        # Test reset
        kg_gen.reset_usage_tracking()
        empty_stats = kg_gen.get_usage_stats()
        assert empty_stats["models"] == {}

    def test_usage_tracking_disabled_by_error(self):
        """Test that usage tracking handles errors gracefully."""
        tracker = UsageTracker()

        # Mock result that raises an exception
        mock_result = Mock()
        mock_result.get_lm_usage.side_effect = Exception("Test error")

        # Should not raise exception
        tracker.track_usage(mock_result, model_name="test-model")

        # Should not have created any stats
        stats = tracker.get_model_stats("test-model")
        assert stats is None

    def test_usage_tracking_no_get_lm_usage_method(self):
        """Test usage tracking with object that doesn't have get_lm_usage method."""
        tracker = UsageTracker()

        # Object without get_lm_usage method
        mock_result = Mock()
        del mock_result.get_lm_usage

        # Should not raise exception
        tracker.track_usage(mock_result, model_name="test-model")

        # Should not have created any stats
        stats = tracker.get_model_stats("test-model")
        assert stats is None


class TestModelStatsToDict:
    """Test ModelStats serialization."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        stats = ModelStats(model_name="test-model")

        usage_data = {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
        stats.add_usage(usage_data)

        result = stats.to_dict()

        assert result["model_name"] == "test-model"
        assert result["total_calls"] == 1
        assert result["total_prompt_tokens"] == 100
        assert result["total_completion_tokens"] == 50
        assert result["total_cost"] == 0.01
        assert "first_call_time" in result
        assert "last_call_time" in result

    def test_to_dict_datetime_serialization(self):
        """Test that datetime objects are properly serialized."""
        stats = ModelStats(model_name="test-model")

        # Add some usage to create timestamps
        stats.add_usage({"prompt_tokens": 10})

        result = stats.to_dict()

        # Check that timestamps are ISO format strings
        assert isinstance(result["first_call_time"], str)
        assert isinstance(result["last_call_time"], str)

        # Verify they can be parsed back as datetime
        first_time = datetime.fromisoformat(result["first_call_time"])
        last_time = datetime.fromisoformat(result["last_call_time"])

        assert isinstance(first_time, datetime)
        assert isinstance(last_time, datetime)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Reset usage tracker before each test."""
        usage_tracker.reset()

    def test_invalid_usage_data_formats(self):
        """Test handling of various invalid usage data formats."""
        tracker = UsageTracker()

        # Test with None
        mock_result = Mock()
        mock_result.get_lm_usage.return_value = None
        tracker.track_usage(mock_result, model_name="test-model")

        # Test with string instead of dict
        mock_result.get_lm_usage.return_value = "invalid data"
        tracker.track_usage(mock_result, model_name="test-model")

        # Test with empty dict
        mock_result.get_lm_usage.return_value = {}
        tracker.track_usage(mock_result, model_name="test-model")

        # Should not have created stats for any of these
        stats = tracker.get_model_stats("test-model")
        assert stats is None

    def test_missing_token_fields(self):
        """Test handling usage data with missing token fields."""
        stats = ModelStats(model_name="test-model")

        # Test with missing fields - should default to 0
        usage_data = {"cost": 0.01}  # Only cost provided
        stats.add_usage(usage_data)

        assert stats.total_calls == 1
        assert stats.total_prompt_tokens == 0
        assert stats.total_completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.total_cost == 0.01

    def test_concurrent_singleton_creation(self):
        """Test that singleton pattern works correctly under concurrent access."""
        instances = []

        def create_instance():
            instances.append(UsageTracker())

        # Create multiple instances concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same object
        assert len(instances) == 10
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance


if __name__ == "__main__":
    pytest.main([__file__])
