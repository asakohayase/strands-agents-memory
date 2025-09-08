import os
import base64
import json
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


def reset_memory(assistant):
    """Clear all stored memories for clean testing"""
    try:
        memories_result = assistant.agent.tool.mem0_memory(
            action="list", user_id=assistant.user_id
        )

        if (
            memories_result.get("status") == "success"
            and memories_result.get("content")
            and len(memories_result["content"]) > 0
        ):
            memories_json = memories_result["content"][0]["text"]
            memories = json.loads(memories_json)

            if memories and len(memories) > 0:
                print(f"🔍 Found {len(memories)} memories to delete")

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


def demonstrate_langfuse_dashboard():
    """LangFuse dashboard visualization with JSON test dataset"""

    if not setup_langfuse_opentelemetry():
        return

    try:
        from strands.telemetry import StrandsTelemetry

        # Configure the telemetry (creates new tracer provider and sets it as global)
        StrandsTelemetry().setup_otlp_exporter()
        print("✅ LangFuse OpenTelemetry integration active")

    except ImportError:
        print(
            "❌ Missing strands-agents[otel] - install with: uv add 'strands-agents[otel]'"
        )
        return

    # Create agent with trace attributes for LangFuse dashboard
    assistant = MovieRecommendationAssistant()

    # Add trace attributes that appear in LangFuse dashboard
    assistant.agent.trace_attributes = {
        "session.id": "synthetic-eval-session-123",
        "user.id": "synthetic-eval-user@example.com",
        "langfuse.tags": [
            "Synthetic-Data",
            "Movie-Agent-Demo",
            "Dashboard-Visualization",
        ],
    }

    # Load test scenarios from JSON file
    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    print("🎬 Running manual movie scenarios...")
    print("All interactions will appear in your LangFuse dashboard\n")

    for scenario in scenarios:
        print(f"Scenario {scenario['scenario_id']}: {scenario['description']}")

        # Always reset memory for clean testing
        reset_memory(assistant)

        # Run each step in the scenario
        for step in scenario["steps"]:
            query = step["user"]
            print(f"Query: {query}")
            response = assistant.agent(query)

        # Run evaluation query
        eval_query = scenario["evaluation_query"]
        print(f"Evaluation Query: {eval_query}")
        response = assistant.agent(eval_query)
        print(f"Agent Response: {response.message}")
        print("-" * 40)

    print("\n🎯 What you can see in LangFuse Dashboard:")
    print("✅ Complete conversation traces for each scenario")
    print("✅ Tool usage breakdown (mem0_memory, rate_movie, recommend_movies)")
    print("✅ Token consumption and cost tracking")
    print("✅ Response time analysis")
    print("✅ Memory operations and retrieval patterns")
    print("✅ Session grouping and filtering by tags")
    print("✅ Agent reasoning steps and decision paths")

    print(
        f"\n🌐 View your traces at: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}"
    )
    print("Navigate to: Traces → Filter by session.id or tags")

    print("\n💡 Dashboard Benefits over Built-in Metrics:")
    print("• Visual timeline of agent execution")
    print("• Historical data across multiple runs")
    print("• Team collaboration and sharing")
    print("• Advanced filtering and search")
    print("• Cost analysis over time")
    print("• Memory operation visibility")


if __name__ == "__main__":
    demonstrate_langfuse_dashboard()
