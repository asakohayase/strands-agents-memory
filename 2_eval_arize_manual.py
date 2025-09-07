"""
eval_arize_synthetic.py - Arize AX with Synthetic Data

Runs synthetic movie scenarios aligned with eval_built_in.py
Setup: uv add arize-otel openinference-instrumentation-bedrock python-dotenv
"""

import os
from dotenv import load_dotenv
from main import MovieRecommendationAssistant

# Load environment variables
load_dotenv()

# Import Arize libraries
try:
    from arize.otel import register
    from openinference.instrumentation.bedrock import BedrockInstrumentor

    ARIZE_AVAILABLE = True
except ImportError:
    ARIZE_AVAILABLE = False
    print(
        "Install: uv add arize-otel openinference-instrumentation-bedrock python-dotenv"
    )


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


def demonstrate_arize_synthetic():
    """Arize AX with synthetic data matching eval_built_in.py"""

    if not ARIZE_AVAILABLE:
        print("Cannot run - missing arize-otel")
        return

    # Get Arize credentials from .env file
    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")

    if not space_id or not api_key:
        print("Error: Set ARIZE_SPACE_ID and ARIZE_API_KEY in .env file")
        return

    # Register with Arize for SYNTHETIC testing
    tracer_provider = register(
        space_id=space_id,
        api_key=api_key,
        project_name="strands-agents-memory-synthetic",  # Separate project for synthetic
    )
    print("Arize AX synthetic testing registered")

    # Add instrumentation AFTER register
    BedrockInstrumentor().instrument(tracer_provider=tracer_provider)
    print("Bedrock instrumentation enabled")

    # Create agent (now automatically traced)
    assistant = MovieRecommendationAssistant()

    # Reset memory for clean synthetic testing
    reset_memory(assistant)

    # Same synthetic test queries as eval_built_in.py
    queries = ["I love Spirited Away", "5", "Recommend movies"]

    for query in queries:
        print(f"Synthetic Query: {query}")

        # Agent execution automatically traced to Arize
        response = assistant.agent(query)

        # Show what's being captured
        print("Sent to Arize (Synthetic):")
        print(f"  Tokens: {response.metrics.accumulated_usage.get('totalTokens', 0)}")
        print(f"  Time: {sum(response.metrics.cycle_durations):.2f}s")
        print(f"  Tools: {list(response.metrics.tool_metrics.keys())}")
        print(f"  Full trace: Agent reasoning + tool calls + responses")

        print()
        print("-" * 40)

    print(f"\nView synthetic traces at: https://app.arize.com")
    print(f"Project: strands-agents-memory-synthetic")


if __name__ == "__main__":
    demonstrate_arize_synthetic()
