#!/usr/bin/env python3
"""
Test script to verify that polarity-aware clustering works correctly.
Tests that positive and negative relations are not clustered together.
"""

import os
import sys

sys.path.insert(0, "src")
from dotenv import load_dotenv

from kg_gen.models import Graph
from kg_gen.kg_gen import KGGen


def test_polarity_preservation():
    """Test that positive and negative relations maintain separate clusters."""

    load_dotenv()

    # Set up model (using GPT-4 as default, can be changed via env var)
    model = "openai/gpt-4.1-nano-2025-04-14"
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    kg_gen = KGGen(model=model, temperature=0.0, api_key=api_key)

    # Create test graph with polarized relations
    test_graph = Graph(
        entities={"company A", "company B", "environment", "economy"},
        edges={
            # Positive relations
            "supports",
            "enhances",
            "promotes",
            "positively impacts",
            "benefits",
            # Negative relations
            "inhibits",
            "reduces",
            "prevents",
            "negatively impacts",
            "harms",
            # Neutral relations
            "affects",
            "influences",
            "relates to",
        },
        relations={
            ("company A", "supports", "environment"),
            ("company A", "enhances", "environment"),
            ("company A", "promotes", "environment"),
            ("company B", "inhibits", "environment"),
            ("company B", "reduces", "environment"),
            ("company B", "prevents", "environment"),
            ("company A", "positively impacts", "economy"),
            ("company B", "negatively impacts", "economy"),
            ("company A", "benefits", "economy"),
            ("company B", "harms", "economy"),
            ("company A", "affects", "economy"),
            ("company A", "influences", "economy"),
            ("company A", "relates to", "economy"),
        },
    )

    print("Original edges:", sorted(test_graph.edges))
    print(f"Original edge count: {len(test_graph.edges)}")
    print()

    # Cluster the graph
    clustered_graph = kg_gen.cluster(
        graph=test_graph,
        context="Trade-offs and synergies between business and environmental impacts",
    )

    print("Clustered edges:", sorted(clustered_graph.edges))
    print(f"Clustered edge count: {len(clustered_graph.edges)}")
    print()

    # Analyze clustering results
    print("Edge clusters:")
    for representative, members in clustered_graph.edge_clusters.items():
        print(f"  {representative}: {sorted(members)}")

    # Check for polarity preservation
    positive_terms = {
        "supports",
        "enhances",
        "promotes",
        "positively impacts",
        "benefits",
    }
    negative_terms = {"inhibits", "reduces", "prevents", "negatively impacts", "harms"}

    polarity_violations = []
    for rep, members in clustered_graph.edge_clusters.items():
        has_positive = any(term in positive_terms for term in members)
        has_negative = any(term in negative_terms for term in members)

        if has_positive and has_negative:
            polarity_violations.append((rep, members))

    print("\nPolarity Analysis:")
    if polarity_violations:
        print("❌ POLARITY VIOLATIONS FOUND:")
        for rep, members in polarity_violations:
            print(f"  Cluster '{rep}' mixes polarities: {sorted(members)}")
    else:
        print(
            "✅ No polarity violations detected - positive and negative relations remain separate"
        )

    return len(polarity_violations) == 0


if __name__ == "__main__":
    success = test_polarity_preservation()
    sys.exit(0 if success else 1)
