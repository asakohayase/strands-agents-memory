import os
import json
import uuid
import pandas as pd
from dotenv import load_dotenv
from main import MovieRecommendationAssistant

load_dotenv()

try:
    from arize.experimental.datasets import ArizeDatasetsClient
    from arize.experimental.datasets.experiments.types import EvaluationResult
    from arize.experimental.datasets.utils.constants import GENERATIVE
    from phoenix.evals import llm_classify, OpenAIModel

    ARIZE_AVAILABLE = True
except ImportError as e:
    ARIZE_AVAILABLE = False
    print(f'Install: pip install "arize[Datasets]" arize-phoenix openai pandas')
    print(f"Error: {e}")


def create_dataset(client, space_id):
    """Create Arize dataset from movie scenarios"""

    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    rows = []
    for scenario in scenarios:
        row = {
            "id": scenario["scenario_id"],
            "description": scenario["description"],
            "steps": json.dumps(scenario["steps"]),
            "evaluation_query": scenario["evaluation_query"],
            "expected_memory": json.dumps(scenario["expected_memory_usage"]),
            "expected_quality": scenario["expected_response_quality"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    dataset_name = f"movie_eval_{uuid.uuid4().hex[:8]}"

    dataset_id = client.create_dataset(
        space_id=space_id,
        dataset_name=dataset_name,
        dataset_type=GENERATIVE,
        data=df,
    )

    print(f"Created dataset: {dataset_name} (ID: {dataset_id})")
    return dataset_id


def movie_task(dataset_row):
    """Execute movie recommendation task"""
    try:
        steps = json.loads(dataset_row.get("steps", "[]"))
        eval_query = dataset_row.get("evaluation_query", "")

        user_id = str(uuid.uuid4())
        assistant = MovieRecommendationAssistant(user_id=user_id)

        # Run scenario steps
        for step in steps:
            assistant.agent(step["user"])

        # Get final response
        result = assistant.agent(eval_query)

        # Extract the actual text from the message structure
        response_text = result.message["content"][0]["text"]

        return response_text

    except Exception as e:
        return f"Task failed: {str(e)}"


def memory_evaluator(output, dataset_row):
    """Evaluate memory utilization using LLM"""

    template = """
    Evaluate if the AI agent correctly used stored user preferences.
    
    Scenario: {description}
    Agent Response: {output}
    Expected Memory Usage: {expected_memory}
    
    Score the memory utilization on a 1-5 scale:
    5 = Perfect use of stored preferences
    4 = Good use with minor gaps
    3 = Partial use of stored preferences  
    2 = Minimal use of stored preferences
    1 = Ignores stored preferences
    
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

        return EvaluationResult(
            score=score, label=f"Memory: {score}/5", explanation=explanation
        )

    except Exception as e:
        return EvaluationResult(
            score=0.0, label="Memory: error", explanation=f"Failed: {str(e)}"
        )


def quality_evaluator(output, dataset_row):
    """Evaluate response quality using LLM"""

    template = """
    Evaluate the quality of movie recommendations.
    
    Agent Response: {output}
    Expected Quality: {expected_quality}
    
    Score the response quality on a 1-5 scale:
    5 = Excellent recommendations, very helpful
    4 = Good recommendations with minor issues
    3 = Adequate recommendations
    2 = Poor recommendations with major issues
    1 = Terrible or unhelpful recommendations
    
    Respond with: 1, 2, 3, 4, or 5
    """

    try:
        expected_quality = dataset_row.get("expected_quality", "")

        df = pd.DataFrame(
            [
                {
                    "output": output,
                    "expected_quality": expected_quality,
                }
            ]
        )

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

        return EvaluationResult(
            score=score, label=f"Quality: {score}/5", explanation=explanation
        )

    except Exception as e:
        return EvaluationResult(
            score=0.0, label="Quality: error", explanation=f"Failed: {str(e)}"
        )


def run_arize_llm_evaluation():
    """Run complete LLM-as-a-judge evaluation with Arize"""

    if not ARIZE_AVAILABLE:
        print("Missing Arize dependencies")
        return

    # Get credentials
    api_key = os.getenv("ARIZE_API_KEY")
    developer_key = os.getenv("ARIZE_DEVELOPER_KEY")
    space_id = os.getenv("ARIZE_SPACE_ID")

    if not (api_key or developer_key) or not space_id:
        print("Missing ARIZE_API_KEY/ARIZE_DEVELOPER_KEY and ARIZE_SPACE_ID")
        return

    # Initialize client
    try:
        if developer_key:
            client = ArizeDatasetsClient(developer_key=developer_key)
        else:
            client = ArizeDatasetsClient(api_key=api_key)
        print("Arize client initialized")
    except Exception as e:
        print(f"Client init failed: {e}")
        return

    # Create dataset
    try:
        dataset_id = create_dataset(client, space_id)
    except Exception as e:
        print(f"Dataset creation failed: {e}")
        return

    # Run separate experiments per scenario
    print(f"\nRunning separate LLM-as-a-judge experiments per scenario...")
    print(f"Dataset ID: {dataset_id}")

    try:
        # Get the dataset
        dataset = client.get_dataset(space_id=space_id, dataset_id=dataset_id)

        experiment_results = []

        # Process each scenario as a separate experiment
        for idx, row in dataset.iterrows():
            scenario_data = row.to_dict()
            scenario_id = scenario_data.get("id", idx + 1)

            print(f"\n=== Running Experiment for Scenario {scenario_id} ===")

            # Create single-row dataset for this scenario
            single_row_df = pd.DataFrame([scenario_data])
            scenario_dataset_name = f"scenario_{scenario_id}_{uuid.uuid4().hex[:8]}"

            scenario_dataset_id = client.create_dataset(
                space_id=space_id,
                dataset_name=scenario_dataset_name,
                dataset_type=GENERATIVE,
                data=single_row_df,
            )

            print(f"Created scenario dataset: {scenario_dataset_name}")

            # Run experiment for this single scenario
            result = client.run_experiment(
                space_id=space_id,
                dataset_id=scenario_dataset_id,
                task=movie_task,
                evaluators=[memory_evaluator, quality_evaluator],
                experiment_name=f"scenario_{scenario_id}_{uuid.uuid4().hex[:8]}",
                exit_on_error=False,
            )

            # Handle tuple result properly
            if isinstance(result, tuple):
                experiment_id, results_df = result
                print(f"Scenario {scenario_id} completed!")
                print(f"Experiment ID: {experiment_id}")
                print(
                    f"Results URL: https://app.arize.com/spaces/{space_id}/experiments/{experiment_id}"
                )

                # Print evaluation results
                if not results_df.empty:
                    memory_score = results_df.iloc[0].get(
                        "eval.memory_evaluator.score", "N/A"
                    )
                    memory_label = results_df.iloc[0].get(
                        "eval.memory_evaluator.label", "N/A"
                    )
                    quality_score = results_df.iloc[0].get(
                        "eval.quality_evaluator.score", "N/A"
                    )
                    quality_label = results_df.iloc[0].get(
                        "eval.quality_evaluator.label", "N/A"
                    )

                    print(f"Final Results: {memory_label} | {quality_label}")

                experiment_results.append(
                    {
                        "scenario_id": scenario_id,
                        "experiment_id": experiment_id,
                        "results_df": results_df,
                    }
                )

            else:
                print(
                    f"Unexpected result type for scenario {scenario_id}: {type(result)}"
                )
                experiment_results.append(
                    {
                        "scenario_id": scenario_id,
                        "experiment_id": None,
                        "error": f"Unexpected result: {result}",
                    }
                )

            print("-" * 60)

        # Print final summary
        print(f"\n=== ALL EXPERIMENTS COMPLETE ===")
        print(f"Successfully ran {len(experiment_results)} separate experiments")

        for exp_result in experiment_results:
            scenario_id = exp_result["scenario_id"]
            exp_id = exp_result["experiment_id"]
            if exp_id:
                print(
                    f"Scenario {scenario_id}: https://app.arize.com/spaces/{space_id}/experiments/{exp_id}"
                )
            else:
                print(
                    f"Scenario {scenario_id}: Failed - {exp_result.get('error', 'Unknown error')}"
                )

        return experiment_results

    except Exception as e:
        print(f"Experiments failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_arize_llm_evaluation()
