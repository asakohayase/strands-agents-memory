import asyncio
from strands import Agent
from strands_tools import mem0_memory, use_llm


USER_ID = "demo_user"

SYSTEM_PROMPT = """You are a movie recommendation assistant with persistent memory capabilities.

Your role is to manage movie preferences and provide personalized recommendations using memory operations.

Key responsibilities:
- Store movie ratings and preferences using mem0_memory tool
- Retrieve relevant user preferences when making recommendations
- Provide personalized movie suggestions based on stored memories
- Learn from user feedback to improve future recommendations

Use the available tools to store, retrieve, and manage user movie preferences effectively.
"""


class MemoryAssistant:
    """Movie recommendation assistant following official Strands memory pattern"""

    def __init__(self, user_id: str = USER_ID):
        self.user_id = user_id
        self.agent = Agent(
            system_prompt=SYSTEM_PROMPT,
            tools=[mem0_memory, use_llm],
            load_tools_from_directory=True,  # Auto-load custom movie tools
        )

        # Initialize with demo memories
        self._initialize_demo_memories()

    def _initialize_demo_memories(self) -> None:
        """Initialize with demo movie preferences"""
        init_memories = "I enjoy sci-fi movies with complex plots and philosophical themes. Christopher Nolan and Denis Villeneuve are among my favorite directors. I loved The Matrix and Inception. I'm not a big fan of horror movies."

        try:
            result = self.agent.tool.mem0_memory(
                action="store", content=init_memories, user_id=self.user_id
            )
            print("✅ Initialized demo memories:", result)
        except Exception as e:
            print(f"⚠️  Could not initialize memories: {e}")

    def process_input(self, user_input: str) -> str:
        """Process user input following official documentation pattern"""
        user_input_lower = user_input.lower()

        # Check if this is a memory storage request
        if user_input_lower.startswith(
            ("remember ", "note that ", "i want you to know ")
        ):
            content = user_input.split(" ", 1)[1] if " " in user_input else user_input

            try:
                result = self.agent.tool.mem0_memory(
                    action="store", content=content, user_id=self.user_id
                )
                print(f"🔍 DEBUG store result: {result}")
                return f"I've stored that information in my memory."
            except Exception as e:
                print(f"❌ ERROR storing memory: {e}")
                return f"Error storing memory: {e}"

        # Check if this is a request to list all memories
        if "show" in user_input_lower and "memories" in user_input_lower:
            try:
                result = self.agent.tool.mem0_memory(
                    action="list", user_id=self.user_id
                )
                print(f"🔍 DEBUG list result: {result}")

                memories = (
                    result.get("memories", []) if isinstance(result, dict) else []
                )
                if memories:
                    memory_list = "\n".join(
                        [
                            f"{i+1}. {mem.get('memory', mem)}"
                            for i, mem in enumerate(memories)
                        ]
                    )
                    return f"Here's everything I remember:\n\n{memory_list}"
                else:
                    return "I don't have any memories stored yet."
            except Exception as e:
                print(f"❌ ERROR listing memories: {e}")
                return f"Error listing memories: {e}"

        # For other queries, retrieve relevant memories and generate response
        try:
            print(f"🔍 DEBUG retrieving memories for query: '{user_input}'")

            retrieved_memories = self.agent.tool.mem0_memory(
                action="retrieve", query=user_input, user_id=self.user_id
            )
            print(f"🔍 DEBUG retrieved memories: {retrieved_memories}")

            memories = (
                retrieved_memories.get("memories", [])
                if isinstance(retrieved_memories, dict)
                else []
            )

            if memories:
                # Format memories for the LLM
                memories_str = "\n".join(
                    [f"- {mem.get('memory', mem)}" for mem in memories]
                )

                prompt = f"""
                User ID: {self.user_id}
                User question: "{user_input}"
                
                Relevant memories for user {self.user_id}:
                {memories_str}
                
                Please generate a helpful movie recommendation response using the memories.
                """

                print(f"🔍 DEBUG calling use_llm with prompt")

                try:
                    response = self.agent.tool.use_llm(
                        prompt=prompt,
                        system_prompt="Generate movie recommendations based on stored user preferences.",
                    )
                    print(f"🔍 DEBUG use_llm response: {response}")

                    # Extract content from response
                    if isinstance(response, dict) and "content" in response:
                        return str(response["content"][0]["text"])
                    else:
                        return str(response)

                except Exception as e:
                    print(f"❌ ERROR with use_llm: {e}")
                    return f"Error generating response: {e}"
            else:
                return "I don't have specific memories about that yet. Tell me about movies you like so I can give you personalized recommendations!"

        except Exception as e:
            print(f"❌ ERROR retrieving memories: {e}")
            return f"Error retrieving memories: {e}"

    async def chat(self, message: str) -> str:
        """Main chat interface"""
        return self.process_input(message)


async def main():
    """CLI interface following official documentation"""
    print("🎬 Movie Recommendation Agent - Direct Tool Usage Pattern")
    print("\nExample Interactions:")
    print("> Remember that I love sci-fi movies")
    print("> I didn't like The Matrix")
    print("> Show me my memories")
    print("> What movies should I watch?")
    print("\nType 'quit' to exit.\n")

    assistant = MemoryAssistant()

    while True:
        try:
            user_input = input("🎬 You: ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Thanks for using the Movie Recommendation Agent! 🎬")
                break

            if not user_input:
                continue

            print("🤖 Agent: ", end="", flush=True)
            response = await assistant.chat(user_input)
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
