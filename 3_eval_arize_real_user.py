"""
eval_arize.py - Arize AX Integration with Strands

Official Arize AX integration for Strands agents.
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


def demonstrate_arize_ax():
    """Official Arize AX integration"""

    if not ARIZE_AVAILABLE:
        print("Cannot run - missing arize-otel")
        return

    # Get Arize credentials from .env file
    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")

    if not space_id or not api_key:
        print("Error: Set ARIZE_SPACE_ID and ARIZE_API_KEY in .env file")
        return

    # Register with Arize (must be BEFORE agent execution)
    tracer_provider = register(
        space_id=space_id, api_key=api_key, project_name="strands-agents-memory"
    )
    print("Arize AX integration registered")

    # Add instrumentation AFTER register
    BedrockInstrumentor().instrument(tracer_provider=tracer_provider)
    print("Bedrock instrumentation enabled")

    # Create agent (now automatically traced)
    assistant = MovieRecommendationAssistant()

    # Test queries
    queries = ["I love Spirited Away", "5", "Recommend movies"]

    for query in queries:
        print(f"Query: {query}")

        # Agent execution automatically traced to Arize
        response = assistant.agent(query)

        # Show what's being captured
        print("Sent to Arize:")
        print(f"  Tokens: {response.metrics.accumulated_usage.get('totalTokens', 0)}")
        print(f"  Time: {sum(response.metrics.cycle_durations):.2f}s")
        print(f"  Tools: {list(response.metrics.tool_metrics.keys())}")
        print(f"  Full trace: Agent reasoning + tool calls + responses")

        print()
        print("-" * 40)

    print(f"\nView traces at: https://app.arize.com")
    print(f"Project: strands-agents-memory")


if __name__ == "__main__":
    demonstrate_arize_ax()
