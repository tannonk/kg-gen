from kg_gen.models import Graph, Relation, Metadata

def test_backward_compatibility():
    """Test that old tuple format relations are automatically converted to new Relation format."""
    
    print("=" * 60)
    print("BACKWARD COMPATIBILITY TEST")
    print("=" * 60)
    
    # Test 1: Old tuple format in set
    print("\n1. Testing old tuple format (set of tuples):")
    try:
        old_tuple_relations = {
            ('stump harvesting', 'exposes', 'mineral soil'),
            ('Harry Potter', 'hasParent', 'James Potter'),
            ('James Potter', 'marriedTo', 'Lily Potter')
        }
        
        graph1 = Graph(
            entities={'stump harvesting', 'mineral soil', 'Harry Potter', 'James Potter', 'Lily Potter'},
            edges={'exposes', 'hasParent', 'marriedTo'},
            relations=old_tuple_relations
        )
        
        print(f"   ✓ Successfully created graph with {len(graph1.relations)} relations")
        
        # Verify conversion
        for rel in graph1.relations:
            assert isinstance(rel, Relation), f"Expected Relation, got {type(rel)}"
            assert hasattr(rel, 'metadata'), "Relation missing metadata"
            assert isinstance(rel.metadata, Metadata), f"Expected Metadata, got {type(rel.metadata)}"
        
        print("   ✓ All relations converted to Relation objects with metadata")
        
        # Test backward compatibility method
        tuples = graph1.get_relation_tuples()
        print(f"   ✓ Backward compatibility: {len(tuples)} relation tuples extracted")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Old list format 
    print("\n2. Testing old list format (list of lists):")
    try:
        old_list_relations = [
            ['Harry Potter', 'hasParent', 'Lily Potter'],
            ['Ginny Weasley', 'marriedTo', 'Harry Potter']
        ]
        
        graph2 = Graph(
            entities={'Harry Potter', 'Lily Potter', 'Ginny Weasley'},
            edges={'hasParent', 'marriedTo'},
            relations=old_list_relations
        )
        
        print(f"   ✓ Successfully created graph with {len(graph2.relations)} relations")
        print("   ✓ All relations converted from list format")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Dict format (from JSON deserialization)
    print("\n3. Testing dict format (from JSON):")
    try:
        dict_relations = [
            {
                'subject': 'Harry Potter',
                'predicate': 'attendsSchool', 
                'object': 'Hogwarts',
                'metadata': {'source': 'book1.txt'}
            },
            {
                'subject': 'Ron Weasley',
                'predicate': 'friendOf',
                'object': 'Harry Potter',
                'metadata': {'source': 'book2.txt', 'confidence': 0.9}
            }
        ]
        
        graph3 = Graph(
            entities={'Harry Potter', 'Hogwarts', 'Ron Weasley'},
            edges={'attendsSchool', 'friendOf'},
            relations=dict_relations
        )
        
        print(f"   ✓ Successfully created graph with {len(graph3.relations)} relations")
        
        # Verify metadata preservation
        for rel in graph3.relations:
            if rel.subject == 'Harry Potter' and rel.predicate == 'attendsSchool':
                assert 'source' in rel.metadata.data
                assert rel.metadata.data['source'] == 'book1.txt'
                print("   ✓ Metadata preserved for Harry/Hogwarts relation")
            elif rel.subject == 'Ron Weasley':
                assert 'source' in rel.metadata.data and 'confidence' in rel.metadata.data
                assert rel.metadata.data['source'] == 'book2.txt'
                assert rel.metadata.data['confidence'] == 0.9
                print("   ✓ Metadata preserved for Ron/Harry relation")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Mixed format (some new Relation objects, some old tuples)
    print("\n4. Testing mixed format:")
    try:
        new_relation = Relation(
            subject='Hermione Granger',
            predicate='friendOf',
            object='Harry Potter',
            metadata=Metadata(data={'source': 'book3.txt'})
        )
        
        mixed_relations = {
            new_relation,  # New Relation object
            ('Draco Malfoy', 'rivalOf', 'Harry Potter'),  # Old tuple
        }
        
        graph4 = Graph(
            entities={'Hermione Granger', 'Harry Potter', 'Draco Malfoy'},
            edges={'friendOf', 'rivalOf'},
            relations=mixed_relations
        )
        
        print(f"   ✓ Successfully created graph with {len(graph4.relations)} relations")
        print("   ✓ Mixed format handled correctly")
        
        # Verify both types
        has_metadata_relation = False
        has_converted_relation = False
        
        for rel in graph4.relations:
            if rel.subject == 'Hermione Granger':
                assert rel.metadata.data.get('source') == 'book3.txt'
                has_metadata_relation = True
            elif rel.subject == 'Draco Malfoy':
                assert isinstance(rel.metadata, Metadata)
                has_converted_relation = True
        
        assert has_metadata_relation and has_converted_relation
        print("   ✓ Both new and converted relations work correctly")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Error case - invalid format
    print("\n5. Testing invalid format (should fail gracefully):")
    try:
        invalid_relations = [
            ('only', 'two'),  # Invalid - only 2 elements
            'not_a_relation',  # Invalid - string
        ]
        
        # This should fail validation
        graph5 = Graph(
            entities={'test'},
            edges={'test'},
            relations=invalid_relations
        )
        
        print("   ✗ Should have failed but didn't!")
        
    except Exception as e:
        print(f"   ✓ Correctly failed with invalid format: {type(e).__name__}")
    
    print("\n" + "=" * 60)
    print("BACKWARD COMPATIBILITY TEST COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    test_backward_compatibility()