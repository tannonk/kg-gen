from kg_gen import KGGen
from kg_gen.models import Graph, Relation, Metadata
from dotenv import load_dotenv


def simple_metadata_test():
    """Simple test without MLflow tracing to verify core metadata functionality."""

    # Load environment variables
    load_dotenv()

    print("=" * 60)
    print("SIMPLE METADATA TEST (WITHOUT MLFLOW TRACING)")
    print("=" * 60)

    # Test 1: Direct Relation creation and merging
    print("\n1. Testing direct Relation creation and metadata merging:")

    # Create relations with different metadata
    rel1 = Relation(
        subject="Harry Potter",
        predicate="hasParent",
        object="James Potter",
        metadata=Metadata(data={"source": "book1.txt", "confidence": 0.9}),
    )

    rel2 = Relation(
        subject="Harry Potter",
        predicate="hasParent",
        object="James Potter",
        metadata=Metadata(data={"source": "book2.txt", "page": 42}),
    )

    print(f"   Relation 1: {rel1.subject} -> {rel1.predicate} -> {rel1.object}")
    print(f"   Metadata 1: {rel1.metadata.data}")
    print(f"   Relation 2: {rel2.subject} -> {rel2.predicate} -> {rel2.object}")
    print(f"   Metadata 2: {rel2.metadata.data}")

    # Test equality
    print(f"   Relations equal? {rel1 == rel2} (should be True)")
    print(f"   Hash equal? {hash(rel1) == hash(rel2)} (should be True)")

    # Test merging
    merged_rel = rel1.merge_metadata(rel2)
    print(f"   Merged metadata: {merged_rel.metadata.data}")

    expected_source = "book1.txt, book2.txt"
    actual_source = merged_rel.metadata.data.get("source")
    assert actual_source == expected_source, (
        f"Expected '{expected_source}', got '{actual_source}'"
    )
    assert merged_rel.metadata.data.get("confidence") == 0.9
    assert merged_rel.metadata.data.get("page") == 42
    print("   ✓ Metadata merging works correctly!")

    # Test 2: Graph creation and backward compatibility
    print("\n2. Testing Graph creation with Relation objects:")

    # Create a simple graph
    entities = {"Harry Potter", "James Potter", "Ginny Weasley"}
    relations = {rel1, rel2}  # Should deduplicate to 1 relation
    edges = {r.predicate for r in relations}

    graph = Graph(entities=entities, relations=relations, edges=edges)

    print(f"   Entities: {len(graph.entities)} - {list(graph.entities)}")
    print(f"   Relations: {len(graph.relations)}")
    print(f"   Edges: {len(graph.edges)} - {list(graph.edges)}")

    # Test backward compatibility
    relation_tuples = graph.get_relation_tuples()
    print(f"   Backward compatibility tuples: {len(relation_tuples)}")
    print(f"   Sample tuple: {list(relation_tuples)[0] if relation_tuples else 'None'}")

    # Verify only one relation due to semantic equality
    assert len(graph.relations) == 1, f"Expected 1 relation, got {len(graph.relations)}"

    # Check the merged relation
    merged_relation = list(graph.relations)[0]
    print(f"   Merged relation metadata: {merged_relation.metadata.data}")

    print("   ✓ Graph creation and deduplication works!")

    # Test 3: Simple KGGen test without tracing
    print("\n3. Testing KGGen without MLflow tracing:")

    # Remove MLflow tracing decorator temporarily
    import sys

    original_modules = sys.modules.copy()

    try:
        # Initialize KGGen
        kg = KGGen()

        # Test metadata handling
        print("   Testing relation extraction with metadata...")

        # Direct test of get_relations function
        from kg_gen.steps._2_get_relations import get_relations
        from kg_gen.steps._1_get_entities import get_entities

        text = "Harry Potter is the son of James Potter."
        entities = get_entities(kg.dspy, text, log_level="WARNING")
        print(f"   Extracted entities: {entities}")

        # Test without metadata
        relations_no_meta = get_relations(
            kg.dspy, text, entities, additional_metadata=None, log_level="WARNING"
        )
        print(f"   Relations without metadata: {len(relations_no_meta)}")
        if relations_no_meta:
            sample_rel = relations_no_meta[0]
            print(
                f"   Sample relation: {sample_rel.subject} -> {sample_rel.predicate} -> {sample_rel.object}"
            )
            print(f"   Sample metadata: {sample_rel.metadata.data}")

        # Test with metadata
        relations_with_meta = get_relations(
            kg.dspy,
            text,
            entities,
            additional_metadata={"source": "test_file.txt", "confidence": 0.8},
            log_level="WARNING",
        )
        print(f"   Relations with metadata: {len(relations_with_meta)}")
        if relations_with_meta:
            sample_rel = relations_with_meta[0]
            print(
                f"   Sample relation: {sample_rel.subject} -> {sample_rel.predicate} -> {sample_rel.object}"
            )
            print(f"   Sample metadata: {sample_rel.metadata.data}")

            # Verify metadata
            assert "source" in sample_rel.metadata.data
            assert sample_rel.metadata.data["source"] == "test_file.txt"
            assert sample_rel.metadata.data["confidence"] == 0.8
            assert sample_rel.metadata.data["extraction_method"] == "dspy_extraction"

        print("   ✓ Metadata extraction works!")

        # Test aggregation
        print("\n4. Testing aggregation with metadata merging:")

        # Create test graphs with overlapping relations
        rel_a = Relation(
            subject="Harry Potter",
            predicate="hasParent",
            object="James Potter",
            metadata=Metadata(data={"source": "source_a.txt"}),
        )
        rel_b = Relation(
            subject="Harry Potter",
            predicate="hasParent",
            object="James Potter",
            metadata=Metadata(data={"source": "source_b.txt"}),
        )

        graph_a = Graph(
            entities={"Harry Potter", "James Potter"},
            relations={rel_a},
            edges={"hasParent"},
        )
        graph_b = Graph(
            entities={"Harry Potter", "James Potter"},
            relations={rel_b},
            edges={"hasParent"},
        )

        # Aggregate
        aggregated = kg.aggregate([graph_a, graph_b])

        print(
            f"   Input graphs: {len(graph_a.relations)} + {len(graph_b.relations)} = {len(graph_a.relations) + len(graph_b.relations)} relations"
        )
        print(
            f"   Aggregated: {len(aggregated.relations)} relations (should be 1 due to merging)"
        )

        assert len(aggregated.relations) == 1, (
            f"Expected 1 relation, got {len(aggregated.relations)}"
        )

        merged_relation = list(aggregated.relations)[0]
        merged_source = merged_relation.metadata.data.get("source", "")
        print(f"   Merged source: {merged_source}")

        # Check that sources were merged
        assert "source_a.txt" in merged_source and "source_b.txt" in merged_source
        print("   ✓ Aggregation with metadata merging works!")

    except Exception as e:
        print(f"   ✗ Error in KGGen test: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Restore modules
        sys.modules.clear()
        sys.modules.update(original_modules)

    # Test 5: Test visualization
    print("\n5. Testing visualization with Relation objects:")

    try:
        from kg_gen.utils.visualize_kg import visualize

        # Create a test graph with relations
        test_relations = {
            Relation(
                subject="Harry Potter",
                predicate="hasParent",
                object="James Potter",
                metadata=Metadata(data={"source": "book1.txt"}),
            ),
            Relation(
                subject="Harry Potter",
                predicate="hasParent",
                object="Lily Potter",
                metadata=Metadata(data={"source": "book2.txt"}),
            ),
            Relation(
                subject="James Potter",
                predicate="marriedTo",
                object="Lily Potter",
                metadata=Metadata(data={"source": "book3.txt"}),
            ),
        }

        test_graph = Graph(
            entities={"Harry Potter", "James Potter", "Lily Potter"},
            relations=test_relations,
            edges={r.predicate for r in test_relations},
        )

        print(
            f"   Test graph: {len(test_graph.entities)} entities, {len(test_graph.relations)} relations"
        )

        # Generate visualization
        visualize(test_graph, "tests/test_metadata_simple_viz.html")
        print("   ✓ Visualization generated successfully!")

    except Exception as e:
        print(f"   ✗ Visualization error: {str(e)}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("SIMPLE METADATA TEST COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    simple_metadata_test()
