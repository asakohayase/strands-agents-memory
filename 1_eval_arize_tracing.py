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


def run_arize_tracing():
    """Arize AX tracing with JSON test dataset"""

    if not ARIZE_AVAILABLE:
        print("Cannot run - missing arize-otel")
        return

    # Get Arize credentials from .env file
    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")

    if not space_id or not api_key:
        print("Error: Set ARIZE_SPACE_ID and ARIZE_API_KEY in .env file")
        return

    # Register with Arize for tracing
    tracer_provider = register(
        space_id=space_id,
        api_key=api_key,
        project_name="strands-agents-memory-tracing",
    )
    print("Arize AX tracing registered")

    # Instrument Bedrock calls to capture LLM interaction traces (prompts, responses, token usage)
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

        # Run each step in the scenario - automatically traced to Arize
        for step in scenario["steps"]:
            user_input = step["user"]
            print(f"\nUser: {user_input}")
            assistant.agent(user_input)

        # Run evaluation query - automatically traced to Arize
        eval_query = scenario["evaluation_query"]
        print(f"\nEvaluation Query: {eval_query}")
        assistant.agent(eval_query)

    print(f"\nView traces at: https://app.arize.com")
    print(f"Project: strands-agents-memory-tracing")


if __name__ == "__main__":
    run_arize_tracing()
