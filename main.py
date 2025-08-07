import asyncio

from strands import Agent
from strands_tools import mem0_memory, use_llm


USER_ID = "user_1"


SYSTEM_PROMPT = """You are a movie recommendation assistant that learns user preferences over time.

Your capabilities:
- Store user movie ratings and preferences in persistent memory using mem0_memory
- Provide personalized movie recommendations based on stored preferences
- Handle natural language expressions like "I didn't like Matrix" or "I loved Inception"
- Understand genre requests like "recommend comedies" or "I want sci-fi movies"

Available genres in database: action, comedy, drama, horror, romance, sci-fi, thriller, fantasy, documentary, animation

Key behaviors:
- When user expresses opinion about a movie, ask for a 1-5 star rating
- Store individual movie ratings and derive genre preferences using rate_movie tool
- When user asks for recommendations, ONLY use the recommend_movies tool - DO NOT generate your own lists or descriptions
- Map user genre requests to available genres:
  * comedies/funny → comedy
  * anime/cartoons → animation  
  * sci-fi/science fiction → sci-fi
  * action movies → action
  * romantic films → romance
  * scary movies → horror
- Present the tool results directly without adding your own movie suggestions
- Remember preferences across conversations

CRITICAL RULE: When calling recommend_movies tool, present its results directly. Do NOT add your own movie lists, descriptions, or duplicate the recommendations. Just use what the tool returns.

Use the tools:
- mem0_memory: Store and retrieve user preferences
- rate_movie: Handle movie rating and series updates  
- recommend_movies: Generate ALL recommendations (present results directly, no additional commentary)
"""


class MovieRecommendationAssistant:
    """Movie recommendation assistant following official Strands memory pattern"""

    def __init__(self, user_id: str = USER_ID):
        self.user_id = user_id
        self.agent = Agent(
            system_prompt=SYSTEM_PROMPT,
            tools=[mem0_memory, use_llm],
            load_tools_from_directory=True,
        )

        try:
            existing = self.agent.tool.mem0_memory(action="list", user_id=self.user_id)
            if existing.get("results"):
                return
        except Exception:
            pass

    def chat(self, message: str) -> str:
        response = self.agent(message)
        return response


async def main():
    print("🎬 Movie Recommendation Agent")
    print("\nExample Interactions:")
    print("> Remember that I love sci-fi movies")
    print("> I didn't like The Matrix")
    print("\nType 'quit' to exit.\n")

    assistant = MovieRecommendationAssistant()

    while True:
        try:
            user_input = input("🎬 You: ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Thanks for using the Movie Recommendation Agent! 🎬")
                break

            if not user_input:
                continue

            print("🤖 Agent: ", end="", flush=True)
            response = assistant.chat(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye! 🎬")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    asyncio.run(main())
