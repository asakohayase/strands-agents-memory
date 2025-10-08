import os
import json
import uuid
import time
from dotenv import load_dotenv
from langfuse import Langfuse
from main import MovieRecommendationAssistant

load_dotenv()


def run_langfuse_llm_evaluation():
    client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    print("Langfuse client initialized")

    with open("movie_evaluation_scenarios.json", "r") as f:
        scenarios = json.load(f)

    dataset_name = f"movie_scenarios_{uuid.uuid4().hex[:8]}"
    client.create_dataset(name=dataset_name)
    print(f"Created dataset: {dataset_name}")

    for scenario in scenarios:
        client.create_dataset_item(
            dataset_name=dataset_name,
            input={
                "input_messages": json.dumps(scenario["input"]),
                "evaluation_query": scenario["evaluation_query"],
            },
            expected_output={"expected_quality": scenario["expected_response_quality"]},
            metadata={
                "scenario_id": scenario["scenario_id"],
                "description": scenario["description"],
            },
        )

    print(f"Uploaded {len(scenarios)} scenarios")
    client.flush()
    time.sleep(2)

    dataset = client.get_dataset(dataset_name)

    print("Running experiment...")
    run_name = f"memory_eval_{uuid.uuid4().hex[:8]}"

    sorted_items = sorted(
        dataset.items,
        key=lambda x: x.metadata.get("scenario_id", 0) if x.metadata else 0,
    )

    for item in sorted_items:
        scenario_id = item.metadata.get("scenario_id") if item.metadata else "unknown"
        print(f"Processing scenario {scenario_id}")

        # Parse input BEFORE item.run()
        input_data = item.input
        input_messages = json.loads(input_data["input_messages"])
        eval_query = input_data["evaluation_query"]
        conversation_text = "\n".join([f"User: {msg}" for msg in input_messages])
        conversation_text += f"\nUser: {eval_query}"

        try:
            # Create agent
            user_id = str(uuid.uuid4())
            assistant = MovieRecommendationAssistant(user_id=user_id)

            # Run all conversation turns OUTSIDE item.run()
            for message in input_messages:
                assistant.agent(message)

            # Get final response OUTSIDE item.run()
            result = assistant.agent(eval_query)
            output_text = result.message["content"][0]["text"]

            # NOW that we have the complete output, use item.run() to link it
            with item.run(run_name=run_name) as root_span:
                # Immediately update the trace with all data
                root_span.update_trace(
                    input=conversation_text,
                    output=output_text,
                    metadata={
                        "scenario_id": scenario_id,
                        "description": (
                            item.metadata.get("description") if item.metadata else None
                        ),
                        "evaluation_query": eval_query,
                        "user_id": user_id,
                    },
                )

            # Flush immediately after context closes
            client.flush()
            time.sleep(3)

            print(f"  ✓ Scenario {scenario_id} complete")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()

    client.flush()

    print(f"\nExperiment complete!")
    print(f"Dataset: {dataset_name}")
    print(f"Run: {run_name}")


if __name__ == "__main__":
    run_langfuse_llm_evaluation()
