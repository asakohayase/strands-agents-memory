import os
import json
import uuid
import pandas as pd
from dotenv import load_dotenv
from main import (
    MovieRecommendationAssistant,
)  # Custom assistant for movie recommendations

# Load environment variables from .env file
load_dotenv()

# Try importing Arize dependencies
try:
    from arize.experimental.datasets import ArizeDatasetsClient
    from arize.experimental.datasets.experiments.types import EvaluationResult
    from arize.experimental.datasets.utils.constants import GENERATIVE
    from phoenix.evals import llm_classify, OpenAIModel
    from openinference.instrumentation import suppress_tracing

    ARIZE_AVAILABLE = True
except ImportError as e:
    ARIZE_AVAILABLE = False
    print('Install: uv add "arize[Datasets]" arize-phoenix openai pandas')
    print(f"Error: {e}")


def movie_task(dataset_row):
    """
    Task function executed for each row in the dataset.
    Input: dataset_row containing scenario steps and evaluation query.
    Output: final response text from the AI assistant.
    """
    try:
        steps = json.loads(dataset_row.get("steps", "[]"))  # Load scenario steps
        eval_query = dataset_row.get(
            "evaluation_query", ""
        )  # Final query for evaluation

        user_id = str(uuid.uuid4())  # Unique ID for this session
        assistant = MovieRecommendationAssistant(user_id=user_id)

        # Execute all steps in the scenario
        for step in steps:
            assistant.agent(step["user"])

        # Execute the final evaluation query
        result = assistant.agent(eval_query)

        # Extract text from the agent's structured message
        response_text = result.message["content"][0]["text"]

        return response_text

    except Exception as e:
        return f"Task failed: {str(e)}"


def memory_evaluator(output, dataset_row):
    """
    Evaluator function to score memory usage.
    Input: output from task and dataset_row.
    Output: EvaluationResult containing score, label, and explanation.
    """
    template = """
    Evaluate if the AI agent correctly used stored user preferences.
    
    Scenario: {description}
    Agent Response: {output}
    Expected Memory Usage: {expected_memory}
    
    Score the memory utilization on a 1-5 scale:
    5 = Perfect use
    4 = Good use
    3 = Partial use
    2 = Minimal use
    1 = Ignored
    
    Respond with: 1, 2, 3, 4, or 5
    """

    try:
        description = dataset_row.get("description", "")
        expected_memory = dataset_row.get("expected_memory", "[]")

        df = pd.DataFrame(
            [
                {
                    "description": description,
                    "output": output,
                    "expected_memory": expected_memory,
                }
            ]
        )

        # Run LLM classification
        with suppress_tracing():
            result = llm_classify(
                data=df,
                template=template,
                model=OpenAIModel(model="gpt-4o-mini"),
                rails=["1", "2", "3", "4", "5"],
                provide_explanation=True,
            )

        label = result["label"][0]
        score = int(label)
        explanation = result.get("explanation", [""])[0]

        return EvaluationResult(score=score, label=str(score), explanation=explanation)

    except Exception as e:
        return EvaluationResult(score=0.0, label="0", explanation=f"Failed: {str(e)}")


def quality_evaluator(output, dataset_row):
    """
    Evaluator function to score response quality.
    Input: output from task and dataset_row.
    Output: EvaluationResult containing score, label, and explanation.
    """
    template = """
    Evaluate the quality of movie recommendations.
    
    Agent Response: {output}
    Expected Quality: {expected_quality}
    
    Score the response quality on a 1-5 scale:
    5 = Excellent
    4 = Good
    3 = Adequate
    2 = Poor
    1 = Terrible
    
    Respond with: 1, 2, 3, 4, or 5
    """

    try:
        expected_quality = dataset_row.get("expected_quality", "")

        df = pd.DataFrame([{"output": output, "expected_quality": expected_quality}])

        with suppress_tracing():
            result = llm_classify(
                data=df,
                template=template,
                model=OpenAIModel(model="gpt-4o-mini"),
                rails=["1", "2", "3", "4", "5"],
                provide_explanation=True,
            )

        label = result["label"][0]
        score = int(label)
        explanation = result.get("explanation", [""])[0]

        return EvaluationResult(score=score, label=str(score), explanation=explanation)

    except Exception as e:
        return EvaluationResult(score=0.0, label="0", explanation=f"Failed: {str(e)}")


def run_arize_llm_evaluation():
    """
    Main function to run LLM-as-a-judge experiments.
    Steps:
    1. Initialize Arize client.
    2. Load scenarios from JSON.
    3. For each scenario:
       - Create dataset.
       - Run task to generate output.
       - Run evaluators to score output.
    4. Collect and print results.
    """
    if not ARIZE_AVAILABLE:
        print("Missing Arize dependencies")
        return

    api_key = os.getenv("ARIZE_API_KEY")
    developer_key = os.getenv("ARIZE_DEVELOPER_KEY")
    space_id = os.getenv("ARIZE_SPACE_ID")

    if not (api_key or developer_key) or not space_id:
        print("Missing ARIZE_API_KEY/ARIZE_DEVELOPER_KEY and ARIZE_SPACE_ID")
        return

    try:
        client = (
            ArizeDatasetsClient(developer_key=developer_key)
            if developer_key
            else ArizeDatasetsClient(api_key=api_key)
        )
        print("Arize client initialized")
    except Exception as e:
        print(f"Client init failed: {e}")
        return

    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    print(f"Running experiments for {len(scenarios)} scenarios...")

    experiment_results = []

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        print(f"\n=== Scenario {scenario_id} ===")

        row = {
            "id": scenario_id,
            "description": scenario["description"],
            "steps": json.dumps(scenario["steps"]),
            "evaluation_query": scenario["evaluation_query"],
            "expected_memory": json.dumps(scenario["expected_memory_usage"]),
            "expected_quality": scenario["expected_response_quality"],
        }

        single_row_df = pd.DataFrame([row])
        scenario_dataset_name = f"scenario_{scenario_id}_{uuid.uuid4().hex[:8]}"

        # Create dataset for this scenario
        scenario_dataset_id = client.create_dataset(
            space_id=space_id,
            dataset_name=scenario_dataset_name,
            dataset_type=GENERATIVE,
            data=single_row_df,
        )
        print(f"Created dataset: {scenario_dataset_name}")

        # Run experiment: generate output and score it
        result = client.run_experiment(
            space_id=space_id,
            dataset_id=scenario_dataset_id,
            task=movie_task,  # Generates output
            evaluators=[memory_evaluator, quality_evaluator],  # Scores output
            experiment_name=f"scenario_{scenario_id}_{uuid.uuid4().hex[:8]}",
            exit_on_error=False,
        )

        if isinstance(result, tuple):
            experiment_id, results_df = result
            print(f"Scenario {scenario_id} completed! Experiment ID: {experiment_id}")
            if not results_df.empty:
                memory_score = results_df.iloc[0].get(
                    "eval.memory_evaluator.score", "N/A"
                )
                quality_score = results_df.iloc[0].get(
                    "eval.quality_evaluator.score", "N/A"
                )
                print(f"Scores: Memory {memory_score}/5 | Quality {quality_score}/5")

            experiment_results.append(
                {
                    "scenario_id": scenario_id,
                    "experiment_id": experiment_id,
                    "results_df": results_df,
                }
            )
        else:
            print(f"Unexpected result type for scenario {scenario_id}: {type(result)}")
            experiment_results.append(
                {
                    "scenario_id": scenario_id,
                    "experiment_id": None,
                    "error": f"Unexpected result: {result}",
                }
            )

        print("-" * 60)

    print(f"All experiments complete: {len(experiment_results)} scenarios processed")
    return experiment_results


if __name__ == "__main__":
    run_arize_llm_evaluation()
