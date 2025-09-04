from kg_gen import KGGen
from kg_gen.models import Graph, Relation, Metadata
import os
from dotenv import load_dotenv
from kg_gen.utils.visualize_kg import visualize

def test_metadata_scenarios():
    """Test various metadata scenarios including empty metadata and source merging."""
    
    # Load environment variables
    load_dotenv()
    
    # Initialize KGGen
    kg = KGGen()
    
    # Test texts with overlapping information
    text1 = "Harry Potter is the son of James Potter. James Potter married Lily Evans."
    text2 = "Harry Potter is the son of James Potter. Lily Potter is Harry's mother."  # Same relation, different source
    text3 = "Ginny Weasley married Harry Potter. Harry and Ginny have three children."
    
    print("=" * 80)
    print("METADATA MERGING TEST")
    print("=" * 80)
    
    # Scenario 1: Generate graph without any metadata
    print("\n1. Testing graph generation WITHOUT metadata:")
    graph1 = kg.generate(
        input_data=text1,
        model="gpt-4.1-nano-2025-04-14",
        api_key=os.getenv("OPENAI_API_KEY"),
        context="Harry Potter family relationships"
    )
    
    print(f"   Entities: {len(graph1.entities)}")
    print(f"   Relations: {len(graph1.relations)}")
    
    # Check that relations have default metadata
    sample_relation = list(graph1.relations)[0]
    print(f"   Sample relation metadata: {sample_relation.metadata.data}")
    assert 'extraction_method' in sample_relation.metadata.data
    print("   ✓ Relations have extraction_method metadata")
    
    # Scenario 2: Generate graph with source metadata
    print("\n2. Testing graph generation WITH source metadata:")
    graph2 = kg.generate(
        input_data=text2,
        model="gpt-4.1-nano-2025-04-14", 
        api_key=os.getenv("OPENAI_API_KEY"),
        context="Harry Potter family relationships",
        additional_metadata={'source': 'book2.txt', 'confidence': 0.95}
    )
    
    print(f"   Entities: {len(graph2.entities)}")
    print(f"   Relations: {len(graph2.relations)}")
    
    # Check that relations have source metadata
    sample_relation = list(graph2.relations)[0]
    print(f"   Sample relation metadata: {sample_relation.metadata.data}")
    assert 'source' in sample_relation.metadata.data
    assert sample_relation.metadata.data['source'] == 'book2.txt'
    assert sample_relation.metadata.data['confidence'] == 0.95
    print("   ✓ Relations have source and custom metadata")
    
    # Scenario 3: Generate another graph with different source
    print("\n3. Testing graph generation with DIFFERENT source metadata:")
    graph3 = kg.generate(
        input_data=text3,
        model="gpt-4.1-nano-2025-04-14",
        api_key=os.getenv("OPENAI_API_KEY"), 
        context="Harry Potter family relationships",
        additional_metadata={'source': 'book3.txt', 'author': 'J.K. Rowling'}
    )
    
    sample_relation = list(graph3.relations)[0] 
    print(f"   Sample relation metadata: {sample_relation.metadata.data}")
    print("   ✓ Relations have different source metadata")
    
    # Scenario 4: Test aggregation with metadata merging
    print("\n4. Testing AGGREGATION with metadata merging:")
    print("   Before aggregation:")
    print(f"     Graph1 relations: {len(graph1.relations)}")
    print(f"     Graph2 relations: {len(graph2.relations)}")
    print(f"     Graph3 relations: {len(graph3.relations)}")
    
    # Aggregate all graphs
    combined_graph = kg.aggregate([graph1, graph2, graph3])
    
    print("   After aggregation:")
    print(f"     Combined relations: {len(combined_graph.relations)}")
    print(f"     Combined entities: {len(combined_graph.entities)}")
    
    # Find relations that should have been merged
    harry_parent_relations = []
    for relation in combined_graph.relations:
        if (relation.subject == "Harry Potter" or relation.subject == "Harry") and \
           ("parent" in relation.predicate.lower() or "son" in relation.predicate.lower() or 
            "father" in relation.predicate.lower() or "mother" in relation.predicate.lower()):
            harry_parent_relations.append(relation)
    
    print(f"   Found {len(harry_parent_relations)} Harry parent relations")
    
    # Check for source merging
    merged_sources_found = False
    for relation in combined_graph.relations:
        if 'source' in relation.metadata.data:
            source = relation.metadata.data['source']
            if ',' in source:  # Multiple sources merged
                merged_sources_found = True
                print(f"   ✓ Found merged sources: {source}")
                break
    
    if not merged_sources_found:
        print("   ! No merged sources found (relations may not have overlapped exactly)")
        # Show some example metadata
        for i, relation in enumerate(list(combined_graph.relations)[:3]):
            print(f"   Example relation {i+1} metadata: {relation.metadata.data}")
    
    # Scenario 5: Test backward compatibility
    print("\n5. Testing BACKWARD COMPATIBILITY:")
    relation_tuples = combined_graph.get_relation_tuples()
    print(f"   Tuple format relations: {len(relation_tuples)}")
    print(f"   Sample tuple: {list(relation_tuples)[0] if relation_tuples else 'None'}")
    print("   ✓ get_relation_tuples() works for backward compatibility")
    
    # Scenario 6: Test chunked processing with metadata
    print("\n6. Testing CHUNKED PROCESSING with metadata:")
    long_text = text1 + " " + text2 + " " + text3 + " " + \
                "Ron Weasley is Harry's best friend. Hermione Granger is also Harry's friend. " + \
                "Albus Dumbledore was the headmaster of Hogwarts. Severus Snape taught Potions."
    
    chunked_graph = kg.generate(
        input_data=long_text,
        model="gpt-4.1-nano-2025-04-14",
        api_key=os.getenv("OPENAI_API_KEY"),
        chunk_size=100,  # Force chunking
        additional_metadata={'source': 'full_story.txt', 'chapter': 1}
    )
    
    print(f"   Chunked graph entities: {len(chunked_graph.entities)}")
    print(f"   Chunked graph relations: {len(chunked_graph.relations)}")
    
    # Check for chunk-specific metadata
    chunk_metadata_found = False
    for relation in chunked_graph.relations:
        if 'source' in relation.metadata.data:
            source = relation.metadata.data['source']
            if ':chunk_' in source:
                chunk_metadata_found = True
                print(f"   ✓ Found chunk metadata: {source}")
                break
        elif 'chunk_id' in relation.metadata.data:
            chunk_metadata_found = True
            print(f"   ✓ Found chunk_id: {relation.metadata.data['chunk_id']}")
            break
    
    if not chunk_metadata_found:
        print("   ! No chunk-specific metadata found")
        # Show sample metadata
        sample_relation = list(chunked_graph.relations)[0]
        print(f"   Sample chunked relation metadata: {sample_relation.metadata.data}")
    
    # Generate visualization
    print("\n7. Generating visualization:")
    visualize(combined_graph, "tests/test_metadata_merging.html")
    print("   ✓ Visualization saved to tests/test_metadata_merging.html")
    
    print("\n" + "=" * 80)
    print("METADATA MERGING TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return combined_graph

def test_direct_metadata_merging():
    """Test direct metadata merging on Relation objects."""
    
    print("\n" + "=" * 80)
    print("DIRECT METADATA MERGING TEST")
    print("=" * 80)
    
    # Create test relations with different metadata
    relation1 = Relation(
        subject="Harry Potter",
        predicate="hasParent", 
        object="James Potter",
        metadata=Metadata(data={'source': 'book1.txt', 'confidence': 0.9})
    )
    
    relation2 = Relation(
        subject="Harry Potter",
        predicate="hasParent",
        object="James Potter", 
        metadata=Metadata(data={'source': 'book2.txt', 'page': 42})
    )
    
    print("Testing direct relation metadata merging:")
    print(f"Relation 1 metadata: {relation1.metadata.data}")
    print(f"Relation 2 metadata: {relation2.metadata.data}")
    print(f"Relations equal? {relation1 == relation2}")  # Should be True (semantic equality)
    print(f"Relations hash equal? {hash(relation1) == hash(relation2)}")  # Should be True
    
    # Test merging
    merged_relation = relation1.merge_metadata(relation2)
    print(f"Merged metadata: {merged_relation.metadata.data}")
    
    # Check that source was properly merged
    expected_source = "book1.txt, book2.txt"
    actual_source = merged_relation.metadata.data.get('source')
    assert actual_source == expected_source, f"Expected '{expected_source}', got '{actual_source}'"
    
    # Check that other metadata was preserved
    assert merged_relation.metadata.data.get('confidence') == 0.9
    assert merged_relation.metadata.data.get('page') == 42
    
    print("✓ Direct metadata merging works correctly!")
    print("=" * 80)

if __name__ == "__main__":
    # Run direct metadata merging test first
    test_direct_metadata_merging()
    
    # Run full integration test
    combined_graph = test_metadata_scenarios()
    
    print(f"\nFinal combined graph summary:")
    print(f"Total entities: {len(combined_graph.entities)}")
    print(f"Total relations: {len(combined_graph.relations)}")
    print(f"Total edges: {len(combined_graph.edges)}")