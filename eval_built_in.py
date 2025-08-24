"""
eval_built_in.py - Strands Built-in Metrics Demo
"""

from main import MovieRecommendationAssistant


def reset_memory(assistant):
    """Clear all stored memories for clean testing"""
    import json

    try:
        memories_result = assistant.agent.tool.mem0_memory(
            action="list", user_id=assistant.user_id
        )

        # Extract the JSON string from the nested structure
        if (
            memories_result.get("status") == "success"
            and memories_result.get("content")
            and len(memories_result["content"]) > 0
        ):

            # Parse the JSON string inside content[0]['text']
            memories_json = memories_result["content"][0]["text"]
            memories = json.loads(memories_json)

            if memories and len(memories) > 0:
                print(f"🔍 Found {len(memories)} memories to delete")

                # Delete each memory
                for memory in memories:
                    assistant.agent.tool.mem0_memory(
                        action="delete",
                        memory_id=memory["id"],
                        user_id=assistant.user_id,
                    )

                print(f"✅ Deleted {len(memories)} memories")
            else:
                print("✅ No memories to delete")
        else:
            print("✅ No memories found")

    except Exception as e:
        print(f"⚠️  Memory reset failed: {e}")


def demonstrate_strands_builtin():
    """Show basic Strands built-in metrics"""

    assistant = MovieRecommendationAssistant()

    reset_memory(assistant)

    # Test queries
    queries = ["I love Spirited Away", "5", "Recommend movies"]

    for query in queries:
        print(f"Query: {query}")
        result = assistant.agent(query)

        # Show the basic metrics you wanted
        print()
        print("-" * 40)
        print(f"Total tokens: {result.metrics.accumulated_usage['totalTokens']}")
        print(f"Execution time: {sum(result.metrics.cycle_durations):.2f} seconds")
        print(f"Tools used: {list(result.metrics.tool_metrics.keys())}")
        print("-" * 40)


if __name__ == "__main__":
    demonstrate_strands_builtin()
