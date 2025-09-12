import os
import base64
import json
import uuid
from dotenv import load_dotenv
from main import MovieRecommendationAssistant

load_dotenv()


def setup_langfuse_opentelemetry():
    """Setup LangFuse"""

    # Get LangFuse credentials
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print("Error: Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env file")
        return False

    # Build Basic Auth header
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    # Configure OpenTelemetry endpoint & headers
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = host + "/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth}"

    print("✅ LangFuse OpenTelemetry environment configured")
    return True


def demonstrate_langfuse_manual():
    """LangFuse dashboard visualization with JSON test dataset"""

    if not setup_langfuse_opentelemetry():
        return

    try:
        from strands.telemetry import StrandsTelemetry

        # Configure the telemetry (creates new tracer provider and sets it as global)
        StrandsTelemetry().setup_otlp_exporter()
        print("LangFuse manual testing registered")

    except ImportError:
        print(
            "❌ Missing strands-agents[otel] - install with: uv add 'strands-agents[otel]'"
        )
        return

    # Load test scenarios from JSON file
    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    print("🎬 Running manual movie scenarios...")
    print("All interactions will appear in your LangFuse dashboard\n")

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO {scenario['scenario_id']}: {scenario['description']}")
        print(f"{'='*60}")

        # Create fresh assistant with unique UUID for each scenario
        user_id = str(uuid.uuid4())
        assistant = MovieRecommendationAssistant(user_id=user_id)

        print(f"Using user_id: {user_id}")

        # Add trace attributes that appear in LangFuse dashboard
        assistant.agent.trace_attributes = {
            "session.id": f"scenario-{scenario['scenario_id']}-{user_id[:8]}",
            "user.id": user_id,
            "langfuse.tags": [
                "Synthetic-Data",
                "Movie-Agent-Demo",
                "Manual-Evaluation",
                f"Scenario-{scenario['scenario_id']}",
            ],
        }

        # Run each step in the scenario
        for step in scenario["steps"]:
            query = step["user"]
            print(f"\nQuery: {query}")
            response = assistant.agent(query)

            # Show metrics
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

        # Show metrics
        print()
        print("-" * 40)
        print(
            f"Total tokens: {response.metrics.accumulated_usage.get('totalTokens', 0)}"
        )
        print(f"Execution time: {sum(response.metrics.cycle_durations):.2f} seconds")
        print(f"Tools used: {list(response.metrics.tool_metrics.keys())}")
        print("-" * 40)


if __name__ == "__main__":
    demonstrate_langfuse_manual()
