from kg_gen import KGGen
from dotenv import load_dotenv

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Initialize KGGen
    kg = KGGen()

    # Test texts
    text1 = "Linda is Joshua's mother. Ben is Josh's brother. Andrew is Josh's father."
    text2 = "Judy is Andrew's sister. Josh is Judy's nephew. Judy is Josh's aunt. Josh also goes by Joshua."

    # Generate individual graphs
    graph1 = kg.generate(
        input_data=text1,
        # model="openai/Qwen/Qwen2.5-7B-Instruct-AWQ",
        model="openai/Qwen/Qwen3-8B-AWQ",
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        context="Family relationships",
        init_kwargs={"max_tokens": 14000},
    )

    # Print results
    print("\nGraph 1:")
    print("Entities:", graph1.entities)
    print("Relations:", graph1.relations)
    print("Edges:", graph1.edges)
    breakpoint()
    graph2 = kg.generate(
        input_data=text2,
        # model="openai/Qwen/Qwen2.5-7B-Instruct-AWQ",
        model="openai/Qwen/Qwen3-8B-AWQ",
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        context="Family relationships",
        init_kwargs={"max_tokens": 14000},
    )

    print("\nGraph 2:")
    print("Entities:", graph2.entities)
    print("Relations:", graph2.relations)
    print("Edges:", graph2.edges)

    # Aggregate the graphs
    combined_graph = kg.aggregate([graph1, graph2])

    print("\nCombined Graph:")
    print("Entities:", combined_graph.entities)
    print("Relations:", combined_graph.relations)
    print("Edges:", combined_graph.edges)

    # Cluster the combined graph
    clustered_graph = kg.cluster(
        combined_graph,
        # model="openai/Qwen/Qwen2.5-7B-Instruct-AWQ",
        model="openai/Qwen/Qwen3-8B-AWQ",
        api_key="EMPTY",
        api_base="http://localhost:8000/v1",
        context="Family relationships",
        init_kwargs={"max_tokens": 14000},
    )

    print("\nClustered Combined Graph:")
    print("Entities:", clustered_graph.entities)
    print("Relations:", clustered_graph.relations)
    print("Edges:", clustered_graph.edges)
    print("Entity Clusters:", clustered_graph.entity_clusters)
    print("Edge Clusters:", clustered_graph.edge_clusters)
