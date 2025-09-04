"""
eval_langfuse_ragas.py - LangFuse + RAGAS Quality Evaluation

Simple demonstration of LangFuse + RAGAS for quality evaluation.
Setup: pip install langfuse ragas datasets pandas
"""

from dotenv import load_dotenv

load_dotenv()

from main import MovieRecommendationAssistant

# Import evaluation libraries
try:
    from langfuse import Langfuse
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("⚠️  Install dependencies: pip install langfuse ragas datasets pandas")


def demonstrate_langfuse_ragas():
    """Simple LangFuse + RAGAS demo"""

    if not DEPS_AVAILABLE:
        print("❌ Cannot run - missing dependencies")
        return

    assistant = MovieRecommendationAssistant()

    # Test queries
    queries = ["I love Spirited Away", "5", "Recommend movies"]

    # Initialize LangFuse (optional - works without API keys for demo)
    try:
        langfuse = Langfuse()
        print("✅ LangFuse connected")
    except:
        langfuse = None
        print("⚠️  LangFuse not configured - skipping tracking")

    for query in queries:
        print(f"Query: {query}")

        # Get agent response
        response = assistant.agent(query)

        # Create evaluation dataset
        dataset = Dataset.from_dict(
            {
                "question": [query],
                "answer": [str(response.message)],
                "contexts": [["User has movie preferences stored in memory"]],
                "ground_truth": ["Should provide relevant movie recommendations"],
            }
        )

        # Run RAGAS evaluation
        try:
            result = evaluate(
                dataset, metrics=[faithfulness, answer_relevancy, context_precision]
            )

            print(f"Faithfulness: {result['faithfulness']:.2f}")
            print(f"Answer Relevancy: {result['answer_relevancy']:.2f}")
            print(f"Context Precision: {result['context_precision']:.2f}")

            # Log to LangFuse if available
            if langfuse:
                trace = langfuse.trace(
                    name="movie_eval", input=query, output=str(response.message)
                )

        except Exception as e:
            print(f"⚠️  RAGAS evaluation failed: {e}")
            print("📊 Faithfulness: 0.85 (mock)")
            print("📊 Answer Relevancy: 0.92 (mock)")
            print("📊 Context Precision: 0.78 (mock)")

        print("-" * 40)


if __name__ == "__main__":
    demonstrate_langfuse_ragas()
