import json
import uuid
from dotenv import load_dotenv
from main import MovieRecommendationAssistant
from strands import Agent

load_dotenv()


def create_evaluator():
    """Create an LLM-as-Judge evaluator agent"""
    evaluator = Agent(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        system_prompt="""
        You are an expert evaluator for movie recommendation agents.
        Score responses on two dimensions (1-5 scale):
        
        Memory Utilization: Does the agent use stored user preferences correctly?
        - 5 = Perfect use of stored preferences
        - 4 = Good use with minor gaps
        - 3 = Partial use of stored preferences  
        - 2 = Minimal use of stored preferences
        - 1 = Ignores stored preferences
        
        Response Quality: Are recommendations accurate and helpful?
        - 5 = Excellent recommendations, very helpful
        - 4 = Good recommendations with minor issues
        - 3 = Adequate recommendations
        - 2 = Poor recommendations with major issues
        - 1 = Terrible or unhelpful recommendations
        
        Return your evaluation in JSON format:
        {
            "memory_score": <1-5>,
            "quality_score": <1-5>,
            "memory_explanation": "<brief explanation>",
            "quality_explanation": "<brief explanation>"
        }
        """,
    )
    return evaluator


def evaluate_response(evaluator, scenario, response):
    """Use LLM-as-Judge to evaluate a response"""
    evaluation_prompt = f"""
    Scenario: {scenario['description']}
    User Context: {scenario['steps']}
    Expected Memory Usage: {scenario['expected_memory_usage']}
    Expected Response Quality: {scenario['expected_response_quality']}
    
    Agent Response: {response}
    
    Evaluate this response on memory utilization and response quality.
    """

    try:
        eval_result = evaluator(evaluation_prompt)

        if isinstance(eval_result.message, dict):
            if "content" in eval_result.message:
                eval_text = eval_result.message["content"][0]["text"]
            else:
                eval_text = str(eval_result.message)
        else:
            eval_text = str(eval_result.message)

        if "```json" in eval_text:
            start = eval_text.find("```json") + 7
            end = eval_text.find("```", start)
            eval_text = eval_text[start:end].strip()
        elif "```" in eval_text:
            start = eval_text.find("```") + 3
            end = eval_text.find("```", start)
            eval_text = eval_text[start:end].strip()

        return json.loads(eval_text)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {
            "memory_score": 0,
            "quality_score": 0,
            "memory_explanation": "Evaluation failed",
            "quality_explanation": "Evaluation failed",
        }


def demonstrate_strands_llm_as_judge():
    """Show Strands built-in LLM-as-Judge evaluation"""

    evaluator = create_evaluator()

    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    results = []

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO {scenario['scenario_id']}: {scenario['description']}")
        print(f"{'='*60}")

        # Create fresh assistant with unique UUID for each scenario
        user_id = str(uuid.uuid4())
        assistant = MovieRecommendationAssistant(user_id=user_id)

        print(f"Using user_id: {user_id}")

        # Run this scenario's steps
        for step in scenario["steps"]:
            query = step["user"]
            print(f"\nQuery: {query}")
            result = assistant.agent(query)

        # Run the evaluation query
        query = scenario["evaluation_query"]
        print(f"\nEvaluation Query: {query}")
        result = assistant.agent(query)

        print("\nEvaluating response with LLM-as-Judge...")
        evaluation = evaluate_response(evaluator, scenario, result.message)

        scenario_result = {
            "scenario_id": scenario["scenario_id"],
            "description": scenario["description"],
            "user_id": user_id,
            "query": query,
            "response": result.message,
            "tokens": result.metrics.accumulated_usage["totalTokens"],
            "execution_time": sum(result.metrics.cycle_durations),
            "tools_used": list(result.metrics.tool_metrics.keys()),
            "memory_score": evaluation["memory_score"],
            "quality_score": evaluation["quality_score"],
            "memory_explanation": evaluation["memory_explanation"],
            "quality_explanation": evaluation["quality_explanation"],
        }
        results.append(scenario_result)

        # LLM-as-a-Judge specific metrics
        print()
        print("-" * 40)
        print("LLM-as-a-Judge Evaluation Results:")
        print(
            f"Memory Score: {evaluation['memory_score']}/5 - {evaluation['memory_explanation']}"
        )
        print(
            f"Quality Score: {evaluation['quality_score']}/5 - {evaluation['quality_explanation']}"
        )
        print(f"Total tokens: {result.metrics.accumulated_usage['totalTokens']}")
        print(f"Execution time: {sum(result.metrics.cycle_durations):.2f} seconds")
        print(f"Tools used: {list(result.metrics.tool_metrics.keys())}")
        print("-" * 40)

    print(f"\n{'='*60}")
    print("LLM-AS-A-JUDGE EVALUATION SUMMARY")
    print(f"{'='*60}")

    if results:
        avg_memory = sum(r["memory_score"] for r in results) / len(results)
        avg_quality = sum(r["quality_score"] for r in results) / len(results)
        total_tokens = sum(r["tokens"] for r in results)
        total_time = sum(r["execution_time"] for r in results)

        print(f"Average Memory Score: {avg_memory:.2f}/5")
        print(f"Average Quality Score: {avg_quality:.2f}/5")
        print(f"Total Tokens: {total_tokens}")
        print(f"Total Time: {total_time:.2f}s")

        print("\nIndividual Results:")
        for result in results:
            print(
                f"   Scenario {result['scenario_id']}: Memory {result['memory_score']}/5, Quality {result['quality_score']}/5"
            )

    return results


if __name__ == "__main__":
    demonstrate_strands_llm_as_judge()
