import os
import json
import uuid
from dotenv import load_dotenv
from main import MovieRecommendationAssistant

# OpenTelemetry imports for manual setup
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from strands_to_openinference_mapping import StrandsToOpenInferenceProcessor

# Load environment variables
load_dotenv()


def run_arize_tracing():
    """Arize AX tracing - official manual OpenTelemetry setup"""

    # Get Arize credentials from .env file
    space_id = os.getenv("ARIZE_SPACE_ID")
    api_key = os.getenv("ARIZE_API_KEY")

    if not space_id or not api_key:
        print("Error: Set ARIZE_SPACE_ID and ARIZE_API_KEY in .env file")
        return

    # Create the Strands to OpenInference processor
    strands_processor = StrandsToOpenInferenceProcessor(debug=True)

    # Create resource with project name
    resource = Resource.create(
        {
            "model_id": "strands-agents-memory-tracing",
            "service.name": "strands-agent-integration",
        }
    )

    # Create tracer provider and add processors
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(strands_processor)

    # Create OTLP exporter for Arize
    otlp_exporter = OTLPSpanExporter(
        endpoint="otlp.arize.com:443",
        headers={"space_id": space_id, "api_key": api_key},
    )
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set the global tracer provider
    trace.set_tracer_provider(provider)

    # Load test scenarios from JSON file
    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO {scenario['scenario_id']}: {scenario['description']}")
        print(f"{'=' * 60}")

        # Create fresh assistant with unique UUID for each scenario
        user_id = str(uuid.uuid4())
        assistant = MovieRecommendationAssistant(user_id=user_id)

        print(f"Using user_id: {user_id}")

        # Set STRANDS_AGENT_SYSTEM_PROMPT for the processor
        os.environ["STRANDS_AGENT_SYSTEM_PROMPT"] = assistant.agent.system_prompt

        # Add trace attributes for better organization in Arize
        assistant.agent.trace_attributes = {
            "session.id": f"scenario-{scenario['scenario_id']}-{user_id[:8]}",
            "user.id": user_id,
            "scenario.id": scenario["scenario_id"],
            "arize.tags": [
                "Agent-SDK",
                "Arize-Project",
                "OpenInference-Integration",
            ],
        }

        # Run each input message in the scenario
        for user_input in scenario["input"]:
            print(f"\nUser: {user_input}")
            assistant.agent(user_input)

        # Run evaluation query
        eval_query = scenario["evaluation_query"]
        print(f"\nEvaluation Query: {eval_query}")
        assistant.agent(eval_query)

    print(f"\nView traces at: https://app.arize.com")
    print(f"Project: strands-agents-memory-tracing")


if __name__ == "__main__":
    run_arize_tracing()
