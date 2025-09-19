import json
import uuid
from dotenv import load_dotenv
from main import MovieRecommendationAssistant
from strands.agent.conversation_manager import SlidingWindowConversationManager

# Load environment variables from .env file
load_dotenv()


def run_strands_tracing():
    """Show basic Strands built-in metrics with isolated scenarios"""

    conversation_manager = SlidingWindowConversationManager(
        should_truncate_results=False  # This will show the full tool result
    )

    # Load scenarios from JSON file
    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO {scenario['scenario_id']}: {scenario['description']}")
        print(f"{'='*60}")

        # Create fresh assistant with unique UUID for each scenario
        user_id = str(uuid.uuid4())
        assistant = MovieRecommendationAssistant(user_id=user_id)
        assistant.agent.conversation_manager = conversation_manager

        print(f"Using user_id: {user_id}")

        # Run this scenario's steps
        for step in scenario["steps"]:
            query = step["user"]
            print(f"Query: {query}")
            result = assistant.agent(query)

            # Show the basic metrics
            print()
            print("-" * 40)
            print(f"Total tokens: {result.metrics.accumulated_usage['totalTokens']}")
            print(f"Execution time: {sum(result.metrics.cycle_durations):.2f} seconds")
            print(f"Tools used: {list(result.metrics.tool_metrics.keys())}")
            print("-" * 40)

        # Run the evaluation query
        query = scenario["evaluation_query"]
        print(f"Query: {query}")
        result = assistant.agent(query)

        # Show the basic metrics
        print()
        print("-" * 40)
        print(f"Total tokens: {result.metrics.accumulated_usage['totalTokens']}")
        print(f"Execution time: {sum(result.metrics.cycle_durations):.2f} seconds")
        print(f"Tools used: {list(result.metrics.tool_metrics.keys())}")
        print("-" * 40)


if __name__ == "__main__":
    run_strands_tracing()
