import os
import base64
import json
import uuid
from dotenv import load_dotenv
from main import MovieRecommendationAssistant

load_dotenv()


def setup_langfuse_opentelemetry():
    """Setup LangFuse OpenTelemetry configuration"""

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

    print("LangFuse OpenTelemetry configured")
    return True


def run_langfuse_tracing():
    """LangFuse tracing with JSON test dataset"""

    if not setup_langfuse_opentelemetry():
        return

    try:
        from strands.telemetry import StrandsTelemetry

        # Configure telemetry to send traces to LangFuse
        StrandsTelemetry().setup_otlp_exporter()
        print("LangFuse tracing registered")

    except ImportError:
        print(
            "Missing strands-agents[otel] - install with: uv add 'strands-agents[otel]'"
        )
        return

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

        # Add trace attributes for LangFuse dashboard organization
        assistant.agent.trace_attributes = {
            "session.id": f"scenario-{scenario['scenario_id']}-{user_id[:8]}",
            "user.id": user_id,
        }

        # Run each step in the scenario - automatically traced to LangFuse
        for step in scenario["steps"]:
            user_input = step["user"]
            print(f"\nUser: {user_input}")
            assistant.agent(user_input)

        # Run evaluation query - automatically traced to LangFuse
        eval_query = scenario["evaluation_query"]
        print(f"\nEvaluation Query: {eval_query}")
        assistant.agent(eval_query)

    print(f"\nView traces at: https://cloud.langfuse.com")


if __name__ == "__main__":
    run_langfuse_tracing()
