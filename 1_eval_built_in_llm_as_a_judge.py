import json
import time
from dotenv import load_dotenv
from main import MovieRecommendationAssistant
from strands import Agent

load_dotenv()


def reset_memory(assistant):
    """Clear all stored memories for clean testing and verify success"""
    import json
    import time

    try:
        print("Starting memory reset...")

        # Step 1: List existing memories
        memories_result = assistant.agent.tool.mem0_memory(
            action="list", user_id=assistant.user_id
        )

        if (
            memories_result.get("status") == "success"
            and memories_result.get("content")
            and len(memories_result["content"]) > 0
        ):
            # Parse the JSON string inside content[0]['text']
            memories_json = memories_result["content"][0]["text"]
            memories = json.loads(memories_json)

            if memories and len(memories) > 0:
                print(f"Found {len(memories)} memories to delete")

                # Delete each memory
                deletion_success = 0
                for i, memory in enumerate(memories):
                    try:
                        delete_result = assistant.agent.tool.mem0_memory(
                            action="delete",
                            memory_id=memory["id"],
                            user_id=assistant.user_id,
                        )
                        if delete_result.get("status") == "success":
                            deletion_success += 1
                        print(
                            f"   Deleted memory {i+1}/{len(memories)}: {memory.get('id', 'NO_ID')}"
                        )
                    except Exception as e:
                        print(f"   Failed to delete memory {i+1}: {e}")

                print(
                    f"Attempted to delete {len(memories)} memories, {deletion_success} reported successful"
                )

                # Step 2: Wait for deletion to propagate
                print("Waiting 2 seconds for deletion to propagate...")
                time.sleep(2)

                # Step 3: Verify deletion by listing again
                print("Verifying memory reset...")
                verification_result = assistant.agent.tool.mem0_memory(
                    action="list", user_id=assistant.user_id
                )

                if (
                    verification_result.get("status") == "success"
                    and verification_result.get("content")
                    and len(verification_result["content"]) > 0
                ):
                    verification_json = verification_result["content"][0]["text"]
                    remaining_memories = json.loads(verification_json)

                    if remaining_memories and len(remaining_memories) > 0:
                        print(
                            f"RESET FAILED: {len(remaining_memories)} memories still exist after deletion!"
                        )
                        for mem in remaining_memories:
                            print(
                                f"   Still exists: {mem.get('id', 'NO_ID')}: {mem.get('memory', 'No content')[:50]}..."
                            )
                        return False
                    else:
                        print("RESET SUCCESSFUL: All memories deleted and verified")
                        return True
                else:
                    print("RESET SUCCESSFUL: No memories found in verification")
                    return True
            else:
                print("No memories to delete")
                return True
        else:
            print("No memories found")
            return True

    except Exception as e:
        print(f"Memory reset failed: {e}")
        import traceback

        traceback.print_exc()
        return False


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

    assistant = MovieRecommendationAssistant()
    evaluator = create_evaluator()

    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    results = []

    print("Starting LLM-as-Judge Evaluation")
    print("=" * 50)

    for scenario in scenarios:
        print(f"\nScenario {scenario['scenario_id']}: {scenario['description']}")

        reset_success = True
        if scenario.get("reset_memory", False):
            print("Resetting memory for clean scenario...")
            reset_success = reset_memory(assistant)

            if not reset_success:
                print(
                    "Memory reset failed - continuing anyway but results may be invalid"
                )
            else:
                print("Memory reset confirmed successful")

            print("Additional 1 second wait after reset...")
            time.sleep(1)

        print("Running scenario setup steps...")
        for step in scenario["steps"]:
            print(f"   Running: {step['user']}")
            try:
                step_result = assistant.agent(step["user"])
                print(f"   Completed")
            except Exception as e:
                print(f"   Failed: {e}")

        query = scenario["evaluation_query"]
        print(f"Evaluation Query: {query}")
        result = assistant.agent(query)

        print("Evaluating response...")
        evaluation = evaluate_response(evaluator, scenario, result.message)

        scenario_result = {
            "scenario_id": scenario["scenario_id"],
            "description": scenario["description"],
            "query": query,
            "response": result.message,
            "tokens": result.metrics.accumulated_usage["totalTokens"],
            "execution_time": sum(result.metrics.cycle_durations),
            "tools_used": list(result.metrics.tool_metrics.keys()),
            "memory_score": evaluation["memory_score"],
            "quality_score": evaluation["quality_score"],
            "memory_explanation": evaluation["memory_explanation"],
            "quality_explanation": evaluation["quality_explanation"],
            "reset_successful": reset_success,
        }
        results.append(scenario_result)

        print(
            f"Memory Score: {evaluation['memory_score']}/5 - {evaluation['memory_explanation']}"
        )
        print(
            f"Quality Score: {evaluation['quality_score']}/5 - {evaluation['quality_explanation']}"
        )
        print(f"Tokens: {result.metrics.accumulated_usage['totalTokens']}")
        print(f"Time: {sum(result.metrics.cycle_durations):.2f}s")

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    if results:
        reset_failures = [r for r in results if not r.get("reset_successful", True)]
        if reset_failures:
            print(
                f"WARNING: {len(reset_failures)} scenarios had memory reset failures:"
            )
            for failure in reset_failures:
                print(
                    f"   - Scenario {failure['scenario_id']}: {failure['description']}"
                )
            print()

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
            reset_status = "✅" if result.get("reset_successful", True) else "❌"
            print(
                f"   {reset_status} Scenario {result['scenario_id']}: Memory {result['memory_score']}/5, Quality {result['quality_score']}/5"
            )

        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to evaluation_results.json")

    return results


if __name__ == "__main__":
    demonstrate_strands_llm_as_judge()
