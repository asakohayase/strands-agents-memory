"""
eval_arize_synthetic.py - Arize AX with Synthetic Data using JSON Test Dataset

Runs synthetic movie scenarios using movie_evaluation_scenarios.json
Setup: uv add arize-otel openinference-instrumentation-bedrock python-dotenv
"""

import os
import json
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


def demonstrate_arize_manual():
    """Arize AX with synthetic data using JSON test dataset"""

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
        project_name="strands-agents-memory-manual",  # Separate project for synthetic
    )
    print("Arize AX Manual testing registered")

    # Add instrumentation AFTER register
    BedrockInstrumentor().instrument(tracer_provider=tracer_provider)
    print("Bedrock instrumentation enabled")

    # Create agent (now automatically traced)
    assistant = MovieRecommendationAssistant()

    # Load test scenarios from JSON file
    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    for scenario in scenarios:
        print(f"Scenario {scenario['scenario_id']}: {scenario['description']}")

        # Always reset memory for clean testing
        reset_memory(assistant)

        # Run each step in the scenario
        for step in scenario["steps"]:
            query = step["user"]
            print(f"Query: {query}")

            # Agent execution automatically traced to Arize
            response = assistant.agent(query)

            # Show what's being captured
            print("Sent to Arize:")
            print(
                f"  Tokens: {response.metrics.accumulated_usage.get('totalTokens', 0)}"
            )
            print(f"  Time: {sum(response.metrics.cycle_durations):.2f}s")
            print(f"  Tools: {list(response.metrics.tool_metrics.keys())}")
            print(f"  Full trace: Agent reasoning + tool calls + responses")

        # Run evaluation query
        eval_query = scenario["evaluation_query"]
        print(f"Evaluation Query: {eval_query}")
        response = assistant.agent(eval_query)

        print("Sent to Arize:")
        print(f"  Tokens: {response.metrics.accumulated_usage.get('totalTokens', 0)}")
        print(f"  Time: {sum(response.metrics.cycle_durations):.2f}s")
        print(f"  Tools: {list(response.metrics.tool_metrics.keys())}")

        print()
        print("-" * 40)

    print(f"\nView traces at: https://app.arize.com")
    print(f"Project: strands-agents-memory-manual")


if __name__ == "__main__":
    demonstrate_arize_manual()
