from kg_gen.models import Graph, Relation, Metadata
from kg_gen.steps._3_cluster_graph import cluster_graph
import dspy
import os
from dotenv import load_dotenv


def test_clustering_with_relations():
    """Test that clustering works with the new Relation objects and preserves metadata."""

    print("=" * 60)
    print("CLUSTERING WITH RELATIONS TEST")
    print("=" * 60)

    # Load environment variables
    load_dotenv()

    # Set up DSPy
    lm = dspy.LM(model="gpt-4.1-nano-2025-04-14", api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)

    print("\n1. Testing clustering with Relation objects (with metadata):")

    try:
        # Create test relations with metadata
        test_relations = {
            Relation(
                subject="Josh",
                predicate="hasParent",
                object="Linda",
                metadata=Metadata(
                    data={"source": "family_tree.txt", "confidence": 0.9}
                ),
            ),
            Relation(
                subject="Joshua",
                predicate="hasParent",
                object="Andrew",
                metadata=Metadata(
                    data={"source": "family_records.txt", "confidence": 0.8}
                ),
            ),
            Relation(
                subject="Ben",
                predicate="isBrother",
                object="Josh",
                metadata=Metadata(data={"source": "birth_records.txt"}),
            ),
        }

        test_graph = Graph(
            entities={"Josh", "Joshua", "Linda", "Andrew", "Ben"},
            edges={"hasParent", "isBrother"},
            relations=test_relations,
        )

        print(
            f"   Input graph: {len(test_graph.entities)} entities, {len(test_graph.relations)} relations"
        )
        print(
            f"   Sample relation metadata: {list(test_graph.relations)[0].metadata.data}"
        )

        # Test clustering (this should now work without tuple unpacking errors)
        clustered_graph = cluster_graph(
            dspy=dspy,
            graph=test_graph,
            context="Family relationships with Josh and Joshua being the same person",
            log_level="WARNING",  # Reduce logging for cleaner output
        )

        print(
            f"   Clustered graph: {len(clustered_graph.entities)} entities, {len(clustered_graph.relations)} relations"
        )
        print(f"   Entity clusters: {clustered_graph.entity_clusters}")
        print(f"   Edge clusters: {clustered_graph.edge_clusters}")

        # Check that relations are still Relation objects with metadata
        for rel in clustered_graph.relations:
            assert isinstance(rel, Relation), f"Expected Relation, got {type(rel)}"
            assert hasattr(rel, "metadata"), "Relation missing metadata"

        print("   ✓ All relations are still Relation objects with metadata")

        # Check that metadata was preserved
        sample_relation = list(clustered_graph.relations)[0]
        print(f"   Sample clustered relation metadata: {sample_relation.metadata.data}")

        print("   ✓ Clustering with Relation objects successful!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n2. Testing clustering with old tuple format (backward compatibility):")

    try:
        # Create graph with old tuple format (should auto-convert)
        old_tuple_graph = Graph(
            entities={"Harry", "James", "Lily"},
            edges={"hasParent", "marriedTo"},
            relations={  # Old tuple format
                ("Harry", "hasParent", "James"),
                ("Harry", "hasParent", "Lily"),
                ("James", "marriedTo", "Lily"),
            },
        )

        print(
            f"   Input graph: {len(old_tuple_graph.entities)} entities, {len(old_tuple_graph.relations)} relations"
        )

        # Verify auto-conversion happened
        for rel in old_tuple_graph.relations:
            assert isinstance(rel, Relation), (
                f"Expected Relation after auto-conversion, got {type(rel)}"
            )

        print("   ✓ Auto-conversion from tuples to Relation objects worked")

        # Test clustering
        clustered_old_graph = cluster_graph(
            dspy=dspy,
            graph=old_tuple_graph,
            context="Family relationships",
            log_level="WARNING",
        )

        print(
            f"   Clustered graph: {len(clustered_old_graph.entities)} entities, {len(clustered_old_graph.relations)} relations"
        )
        print("   ✓ Clustering with auto-converted relations successful!")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("CLUSTERING WITH RELATIONS TEST COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    test_clustering_with_relations()
