import os
import json
import uuid
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
        project_name="strands-agents-memory-manual",
    )
    print("Arize AX built_in_manual testing registered")

    # Add instrumentation AFTER register
    BedrockInstrumentor().instrument(tracer_provider=tracer_provider)
    print("Bedrock instrumentation enabled")

    # Load test scenarios from JSON file
    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO {scenario['scenario_id']}: {scenario['description']}")
        print(f"{'='*60}")

        # Create fresh assistant with unique UUID for each scenario
        user_id = str(uuid.uuid4())
        assistant = MovieRecommendationAssistant(user_id=user_id)

        print(f"Using user_id: {user_id}")

        # Run each step in the scenario
        for step in scenario["steps"]:
            query = step["user"]
            print(f"\nQuery: {query}")

            # Agent execution automatically traced to Arize
            response = assistant.agent(query)

            # Show metrics - consistent with built_in_manual format
            print()
            print("-" * 40)
            print(
                f"Total tokens: {response.metrics.accumulated_usage.get('totalTokens', 0)}"
            )
            print(
                f"Execution time: {sum(response.metrics.cycle_durations):.2f} seconds"
            )
            print(f"Tools used: {list(response.metrics.tool_metrics.keys())}")
            print("-" * 40)

        # Run evaluation query
        eval_query = scenario["evaluation_query"]
        print(f"\nQuery: {eval_query}")
        response = assistant.agent(eval_query)

        # Show metrics - consistent with built_in_manual format
        print()
        print("-" * 40)
        print(
            f"Total tokens: {response.metrics.accumulated_usage.get('totalTokens', 0)}"
        )
        print(f"Execution time: {sum(response.metrics.cycle_durations):.2f} seconds")
        print(f"Tools used: {list(response.metrics.tool_metrics.keys())}")
        print("-" * 40)

    print(f"\nView traces at: https://app.arize.com")
    print(f"Project: strands-agents-memory-manual")


if __name__ == "__main__":
    demonstrate_arize_manual()
