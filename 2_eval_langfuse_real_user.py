"""
eval_langfuse_real_user.py - LangFuse Dashboard with Real User Input

Interactive console that captures real user interactions for LangFuse dashboard
"""

import os
import base64
import uuid
import asyncio
from dotenv import load_dotenv
from main import MovieRecommendationAssistant

load_dotenv()


def setup_langfuse_opentelemetry():
    """Setup LangFuse with OpenTelemetry following official docs"""

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

    return True


class DashboardMovieAssistant:
    """Movie assistant that captures real user interactions for LangFuse dashboard"""

    def __init__(self, user_id="real_user_1"):
        self.assistant = MovieRecommendationAssistant(user_id=user_id)
        self.interaction_count = 0

        # Setup OpenTelemetry tracing
        try:
            from strands.telemetry import StrandsTelemetry

            strands_telemetry = StrandsTelemetry().setup_otlp_exporter()
            print("✅ LangFuse OpenTelemetry integration configured")
        except ImportError:
            print("❌ Missing strands-agents[otel]")
            return

        # Add trace attributes for real user data
        session_id = "real-user-session-" + str(uuid.uuid4())[:8]
        self.assistant.agent.trace_attributes = {
            "session.id": session_id,
            "user.id": "real-user@example.com",
            "langfuse.tags": [
                "Real-User-Data",
                "Interactive-Console",
                "Movie-Recommendations",
            ],
        }

        print(f"📊 Dashboard session ID: {session_id}")

    def chat_with_dashboard_tracking(self, user_input: str):
        """Process real user input with dashboard tracking"""

        self.interaction_count += 1

        # Get agent response (automatically traced to LangFuse via OpenTelemetry)
        response = self.assistant.agent(user_input)

        # Show what's being captured for the dashboard
        print(f"\n📈 Interaction #{self.interaction_count} captured in LangFuse:")
        print(f"  ├─ Input: {user_input}")
        print(
            f"  ├─ Tokens: {response.metrics.accumulated_usage.get('totalTokens', 0)}"
        )
        print(f"  ├─ Response Time: {sum(response.metrics.cycle_durations):.2f}s")
        print(
            f"  ├─ Tools Used: {', '.join(response.metrics.tool_metrics.keys()) if response.metrics.tool_metrics else 'None'}"
        )
        print(f"  └─ Full trace with agent reasoning")

        return response

    def show_dashboard_summary(self):
        """Show summary of what's available in the dashboard"""

        if self.interaction_count == 0:
            print("\n📊 No interactions captured yet")
            return

        print(f"\n🎯 LangFuse Dashboard Summary:")
        print(f"✅ {self.interaction_count} real user interactions captured")
        print(f"✅ Complete conversation history with agent reasoning")
        print(f"✅ Tool usage patterns and memory operations")
        print(f"✅ Performance metrics (tokens, timing)")
        print(f"✅ Session grouping for analysis")

        print(
            f"\n🌐 View at: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}"
        )
        print(
            f"Filter by session.id: {self.assistant.agent.trace_attributes['session.id']}"
        )


async def main():
    """Interactive console for real user data collection with dashboard tracking"""

    print("🎬 Movie Recommendation Agent - LangFuse Dashboard Demo")
    print(
        "\nYour real interactions will be captured and visualized in LangFuse dashboard"
    )
    print("Type 'summary' to see dashboard overview")
    print("Type 'quit' to exit\n")

    if not setup_langfuse_opentelemetry():
        return

    assistant = DashboardMovieAssistant()

    while True:
        try:
            user_input = input("🎬 You: ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                assistant.show_dashboard_summary()
                print("\nThanks for providing real user data for the dashboard!")
                break

            if user_input.lower() == "summary":
                assistant.show_dashboard_summary()
                continue

            if not user_input:
                continue

            print("🤖 Agent: ", end="", flush=True)
            response = assistant.chat_with_dashboard_tracking(user_input)
            print(response)

        except KeyboardInterrupt:
            print(f"\n\nDashboard captured {assistant.interaction_count} interactions")
            assistant.show_dashboard_summary()
            print("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    asyncio.run(main())
