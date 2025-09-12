from kg_gen import KGGen
import os
from dotenv import load_dotenv

from kg_gen.utils.visualize_kg import visualize

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Example usage
    kg = KGGen()

    # Reset usage tracking to start fresh
    kg.reset_usage_tracking()

    # Generate a simple graph
    text = "Harry has two parents - his dad James Potter and his mom Lily Potter. Harry and his wife Ginny have three kids together: their oldest son James Sirius, their other son Albus, and their daughter Lily Luna."

    # graph = kg.generate(
    #   input_data=text,
    #   model="openai/Qwen/Qwen3-8B-AWQ",
    #   api_key="EMPTY",
    #   api_base="http://localhost:8000/v1"
    # )

    graph = kg.generate(
        input_data=text,
        model="openai/gpt-4.1-nano-2025-04-14",
        api_key=os.getenv("OPENAI_API_KEY"),
        context="Family relationships",
    )

    visualize(graph, "tests/test_basic.html")

    # Demonstrate usage tracking functionality
    print("\n" + "=" * 50)
    print("USAGE TRACKING DEMONSTRATION")
    print("=" * 50)

    # Print usage summary
    print("\nUsage Summary:")
    print(kg.get_usage_summary())

    # Get detailed stats
    stats = kg.get_usage_stats()
    print("\nDetailed Stats:")
    print(f"Models used: {list(stats['models'].keys())}")
    print(f"Total API calls: {stats['aggregate'].get('total_calls', 0)}")
    print(f"Total tokens: {stats['aggregate'].get('total_tokens', 0):,}")
    print(f"Total cost: ${stats['aggregate'].get('total_cost', 0.0):.4f}")

    # Export usage data
    print("\nExporting usage data...")
    json_export = kg.export_usage_json()
    print(f"JSON export length: {len(json_export)} characters")

    # Save to CSV
    csv_filename = "tests/usage_stats_basic.csv"
    kg.export_usage_csv(csv_filename)
    print(f"Usage stats saved to: {csv_filename}")

    print("\nUsage tracking demonstration complete!")
    print("=" * 50)

    # print(graph)
