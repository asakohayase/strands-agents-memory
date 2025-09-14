import os
import json
import uuid
import time
from datetime import datetime
from dotenv import load_dotenv
from main import MovieRecommendationAssistant

load_dotenv()

try:
    from langfuse import observe, get_client

    LANGFUSE_AVAILABLE = True
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    print(f"Install: pip install langfuse")
    print(f"Error: {e}")


class LangfuseLLMAsJudge:
    """
    Langfuse built-in LLM-as-a-Judge implementation for movie recommendation evaluation.
    """

    def __init__(self):
        if not LANGFUSE_AVAILABLE:
            raise ImportError(
                "Langfuse not available. Install with: pip install langfuse"
            )

        self.langfuse = get_client()
        self.session_id = f"movie_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print("🎬 Langfuse LLM-as-a-Judge Evaluation Setup")
        print("=" * 60)
        print("Creating traces for evaluation scenarios...")
        print("Set up evaluators in Langfuse dashboard to score traces automatically")

    def run_scenario_with_tracing(self, scenario):
        """Run a single scenario while creating proper Langfuse traces"""
        scenario_id = scenario["scenario_id"]
        description = scenario["description"]
        steps = scenario["steps"]
        evaluation_query = scenario["evaluation_query"]

        print(f"\n{'='*60}")
        print(f"SCENARIO {scenario_id}: {description}")
        print(f"{'='*60}")

        # Create fresh assistant with unique user ID
        user_id = str(uuid.uuid4())
        assistant = MovieRecommendationAssistant(user_id=user_id)

        print(f"Using user_id: {user_id}")

        # Create trace using context manager
        with self.langfuse.start_as_current_span(
            name=f"movie_recommendation_eval_scenario_{scenario_id}",
            input={"evaluation_query": evaluation_query},
        ) as root_span:

            # Set trace attributes
            root_span.update_trace(
                session_id=self.session_id,
                user_id=user_id,
                metadata={
                    "scenario_id": scenario_id,
                    "description": description,
                    "evaluation_type": "llm_as_judge",
                    "expected_memory_usage": scenario["expected_memory_usage"],
                    "expected_response_quality": scenario["expected_response_quality"],
                },
                tags=["evaluation", "movie_recommendation", f"scenario_{scenario_id}"],
            )

            start_time = time.time()

            # Execute scenario setup steps
            for i, step in enumerate(steps):
                query = step["user"]
                print(f"\nSetup Step {i+1}: {query}")

                # Create span for setup step
                with root_span.start_as_current_span(
                    name=f"setup_step_{i+1}", input=query
                ) as setup_span:
                    setup_span.update(
                        metadata={"expected_memory": step["expected_memory"]}
                    )

                    result = assistant.agent(query)

                    # Extract response text
                    if isinstance(result.message, dict):
                        if "content" in result.message:
                            response_text = result.message["content"][0]["text"]
                        else:
                            response_text = str(result.message)
                    else:
                        response_text = str(result.message)

                    setup_span.update(output=response_text)

            # Execute the main evaluation query
            print(f"\nEvaluation Query: {evaluation_query}")

            with root_span.start_as_current_generation(
                name="movie_recommendation_response", input=evaluation_query
            ) as generation:
                generation.update(
                    metadata={"evaluation_query": True, "scenario_context": scenario}
                )

                result = assistant.agent(evaluation_query)
                execution_time = time.time() - start_time

                # Extract response text
                if isinstance(result.message, dict):
                    if "content" in result.message:
                        response_text = result.message["content"][0]["text"]
                    else:
                        response_text = str(result.message)
                else:
                    response_text = str(result.message)

                # Update generation with response and usage
                generation.update(
                    output=response_text,
                    usage={
                        "input_tokens": result.metrics.accumulated_usage.get(
                            "inputTokens", 0
                        ),
                        "output_tokens": result.metrics.accumulated_usage.get(
                            "outputTokens", 0
                        ),
                        "total_tokens": result.metrics.accumulated_usage["totalTokens"],
                    },
                )

            # Update trace with final results
            root_span.update_trace(
                output=response_text,
                metadata={
                    "tokens": result.metrics.accumulated_usage["totalTokens"],
                    "execution_time_seconds": execution_time,
                    "tools_used": list(result.metrics.tool_metrics.keys()),
                },
            )

            print(f"\nResponse: {response_text}")
            print(f"Tokens: {result.metrics.accumulated_usage['totalTokens']}")
            print(f"Execution time: {execution_time:.2f}s")
            print(f"Tools used: {list(result.metrics.tool_metrics.keys())}")
            print(f"Trace ID: {root_span.trace_id}")

            return {
                "scenario_id": scenario_id,
                "user_id": user_id,
                "query": evaluation_query,
                "response": response_text,
                "tokens": result.metrics.accumulated_usage["totalTokens"],
                "execution_time": execution_time,
                "tools_used": list(result.metrics.tool_metrics.keys()),
                "trace_id": root_span.trace_id,
                "description": description,
            }

    def run_all_scenarios(self):
        """Run all evaluation scenarios with proper tracing"""
        with open("movie_evaluation_scenarios.json", "r") as f:
            scenarios = json.load(f)

        print(f"\n🚀 Running {len(scenarios)} evaluation scenarios")
        print(f"Session ID: {self.session_id}")

        results = []

        for scenario in scenarios:
            scenario_result = self.run_scenario_with_tracing(scenario)
            results.append(scenario_result)

        self._print_summary(results)
        return results

    def _print_summary(self, results):
        """Print evaluation summary and dashboard instructions"""
        print(f"\n{'='*60}")
        print("LANGFUSE EVALUATION SUMMARY")
        print(f"{'='*60}")

        if results:
            total_tokens = sum(r["tokens"] for r in results)
            total_time = sum(r["execution_time"] for r in results)

            print(f"Session ID: {self.session_id}")
            print(f"Scenarios completed: {len(results)}")
            print(f"Total Tokens: {total_tokens}")
            print(f"Total Time: {total_time:.2f}s")

            print("\nTrace IDs:")
            for result in results:
                print(f"   Scenario {result['scenario_id']}: {result['trace_id']}")

        print(f"\n{'='*60}")
        print("NEXT STEPS: SET UP EVALUATORS IN LANGFUSE DASHBOARD")
        print(f"{'='*60}")

        dashboard_url = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        print(f"1. Go to your Langfuse dashboard: {dashboard_url}")
        print("2. Navigate to 'Evaluators' page")
        print("3. Click '+ Set up Evaluator'")
        print("4. Set up these two evaluators:")

        print("\n📊 MEMORY UTILIZATION EVALUATOR:")
        print("   Name: Memory_Utilization")
        print("   Variable mappings:")
        print("     {{input}} → $.input")
        print("     {{expected_memory_usage}} → $.metadata.expected_memory_usage")
        print("     {{output}} → $.output")

        print("\n🎯 RESPONSE QUALITY EVALUATOR:")
        print("   Name: Response_Quality")
        print("   Variable mappings:")
        print("     {{input}} → $.input")
        print(
            "     {{expected_response_quality}} → $.metadata.expected_response_quality"
        )
        print("     {{output}} → $.output")

        print(f"\n5. Configure filters to run on traces with:")
        print(f"   - Session ID: {self.session_id}")
        print(f"   - Tags: evaluation, movie_recommendation")
        print(f"   - Trace name: movie_recommendation_eval_scenario_*")

        print("\n6. The evaluators will automatically score your traces!")
        print("7. View results in the 'Scores' dashboard")

    def get_dashboard_setup_info(self):
        """Get structured information for dashboard setup"""
        return {
            "session_id": self.session_id,
            "dashboard_url": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            "filters": {
                "session_id": self.session_id,
                "tags": ["evaluation", "movie_recommendation"],
                "trace_name_pattern": "movie_recommendation_eval_scenario_*",
            },
            "evaluators": [
                {
                    "name": "Memory_Utilization",
                    "variables": ["input", "expected_memory_usage", "output"],
                    "score_range": "1-5",
                }
            ],
        }


def demonstrate_langfuse_official_llm_as_judge():
    """Main demonstration following official Langfuse approach"""
    if not LANGFUSE_AVAILABLE:
        print("Langfuse not available. Please install with: pip install langfuse")
        return []

    # Initialize evaluator
    evaluator = LangfuseLLMAsJudge()

    # Run all scenarios (creates traces)
    results = evaluator.run_all_scenarios()

    # Flush to ensure all traces are sent
    evaluator.langfuse.flush()

    return results


if __name__ == "__main__":
    demonstrate_langfuse_official_llm_as_judge()
