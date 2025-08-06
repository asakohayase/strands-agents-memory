# Movie Recommendation Assistant 🎥

> A movie recommendation agent showcasing Strands memory capabilities with persistent learning and personalized recommendations.  
This project demonstrates how to build an AI agent that learns user preferences over time using the Strands Agents SDK and Mem0 memory system.


## 1. Features

- **Persistent Memory**: Learns and remembers user preferences across conversations  
- **Personalized Recommendations**: Tailored movie suggestions based on stored preferences  
- **Natural Language Processing**: Understands expressions like “I didn’t like Matrix” or “I loved Inception”  
- **Genre Intelligence**: Maps user requests to available genres and learns preferences  
- **Series Awareness**: Automatically applies ratings to entire movie series when applicable  

## 2. Architecture

The system combines Strands Agents with Mem0's memory capabilities:

- **Strands Agent**: Core conversational AI with tool integration  
- **Mem0 Memory**: Memory operations (store, retrieve) and preference extraction
- **FAISS Vector Database**: Local persistent storage for memory embeddings 
- **Custom Tools**: Specialized movie rating and recommendation tools  
- **Movie Database**: Curated collection of movies with genres and series information  
- **Amazon Bedrock**: Claude 3.7 Sonnet model for natural language understanding  


## 3. Memory & Recommendation Flow
The movie recommendation agent uses different tool combinations depending on the user's request. Here are the detailed flows:

### Comedy Recommendation Flow
```text
User: "recommend comedy movies"
    ↓
Strands Agent: Process Query & Plan Reasoning
    ↓
Execute Tools: recommend_movies tool
    ↓
recommend_movies tool:
  1. Calls Mem0 Memory (action="retrieve", query="comedy preferences")
  2. Mem0 internally:
     - Uses Claude 3.5 Haiku for memory processing
     - Searches FAISS vector store for relevant memories
     - Returns user preference data
  3. Accesses Movie Database (Python data structure)
  4. Applies scoring algorithm based on:
     - User's genre preferences from memory
     - Movie ratings and genres
     - Preference-based boosts/penalties
  5. Returns top-rated comedy recommendations
    ↓
Strands Agent: Generate Response using recommendations
```

### Movie Rating Flow
```text
User: "I rate Inception 5 stars"
    ↓
Strands Agent: Process Query & Plan Reasoning  
    ↓
Execute Tools: rate_movie tool
    ↓
rate_movie tool:
  1. Searches Movie Database for "Inception"
  2. Finds matching movie and any series information
  3. Creates structured memory content:
     - "User rated 'Inception' 5/5 stars"
     - "User liked this movie"
     - "Genres: sci-fi, thriller"
     - Genre preference: "User likes sci-fi, thriller movies"
  4. Calls Mem0 Memory (action="store", content=rating_info)
  5. Mem0 processes and stores in FAISS vector database
  6. Returns success confirmation
    ↓
Strands Agent: Generate confirmation response
```

## 4. Quick Start

### Prerequisites

- Python 3.10+  
- AWS account with Bedrock access (see setup below)  
- *(Optional)* API keys for cloud memory backends (Qdrant, OpenSearch)  


### AWS Bedrock Setup

#### 1. Create AWS Account  
Go to [aws.amazon.com](https://aws.amazon.com) and create an account.

#### 2. Enable Bedrock Access  

- Navigate to **Amazon Bedrock** in the AWS Console  
- Go to **Model access**
- Request access to the following models:
  - Claude 3.7 Sonnet (us.anthropic.claude-3-7-sonnet-20250219-v1:0) - Used by your main agent
  - Claude 3.5 Haiku (anthropic.claude-3-5-haiku-20241022-v1:0) - Used by Mem0 for fast fact extraction
  - Amazon Titan Text Embeddings v2 (amazon.titan-embed-text-v2:0) - Used by Mem0 for vector embeddings 
- Wait for approval (usually instant for most accounts)

#### 3. Create IAM User with Bedrock Permissions

1. Go to **IAM** service in the AWS Console  
2. Click **Users** → **Create user**  
3. Enter a username and click **Next**  
4. Select **Attach policies directly**  
5. Search for and select **AmazonBedrockFullAccess**  
6. Complete user creation  
7. Go to the new user and click **Create access key**  
8. Choose **Application running outside AWS** and create keys  
9. Download or copy the **Access Key ID** and **Secret Access Key**

#### 4. Configure AWS Credentials

```bash
aws configure
# Enter your Access Key ID
# Enter your Secret Access Key
# Enter region: us-west-2
```

### Installation
1. Clone the repository
```bash
git clone https://github.com/asakohayase/strands-agents-memory.git
cd strands-agents-memory
```

2. Install dependencies
```
uv sync
```

3. Run the Assistant
```bash
uv run python main.py
```

📋 Usage Examples
```text
🎬 You: I love sci-fi movies with complex plots  
🤖 Agent: I'll remember that you enjoy sci-fi movies with complex plots...

🎬 You: I didn't like The Matrix  
🤖 Agent: What would you rate The Matrix on a scale of 1-5 stars?

🎬 You: 2 stars  
🤖 Agent: Thanks! I've stored your rating...

🎬 You: Recommend some comedies  
🤖 Agent: Based on your preferences, here are some comedy recommendations...
```

## 5. Core Implementation
Movie Recommendation Assistant Setup
```python
from strands import Agent
from strands_tools import mem0_memory, use_llm

class MovieRecommendationAssistant:
    def __init__(self, user_id: str = "demo_user"):
        self.agent = Agent(
            system_prompt=SYSTEM_PROMPT,
            tools=[mem0_memory, use_llm],
            load_tools_from_directory=True,
            model=bedrock_model,
        )
```
Rate Movie Tool
```python
@tool
def rate_movie(movie_title: str, user_rating: float, liked: bool) -> Dict[str, Any]:
    matched_movie = get_movie_by_title(movie_title)
    movies_to_rate = [matched_movie]
    if matched_movie.series:
        movies_to_rate = get_movies_by_series(matched_movie.series)
    
    memory_entries = []
    for movie in movies_to_rate:
        memory_content = f"User rated '{movie.title}' {user_rating}/5 stars..."
        memory_entries.append(memory_content)
```
Recommend Movies Tool
```python
@tool
def recommend_movies(user_memories: str = "", count: int = 5, genre_filter: str = None) -> Dict[str, Any]:
    all_movies = get_all_movies()
    for movie in filtered_movies:
        score = movie.rating / 10.0
        # Apply preference boosts/penalties
    # Return sorted recommendations
```
Movie Database Structure
```python
@dataclass
class Movie:
    id: str
    title: str
    year: int
    genres: List[Genre]
    rating: float
    series: str = None

class Genre(str, Enum):
    ACTION = "action"
    COMEDY = "comedy"
    DRAMA = "drama"
    HORROR = "horror"
    ROMANCE = "romance"
    SCI_FI = "sci-fi"
    THRILLER = "thriller"
    FANTASY = "fantasy"
    DOCUMENTARY = "documentary"
    ANIMATION = "animation"
```
## 6. Links
- Mem0 Memory Agent Official Documentation: https://strandsagents.com/latest/documentation/docs/examples/python/memory_agent/
- Mem0 Memory Tool Github Repo: https://github.com/strands-agents/tools/blob/main/src/strands_tools/mem0_memory.py

## 7. Contributing

We welcome contributions! Whether it's a bug fix, new feature, or suggestion for improvement — every bit helps.

### How to Contribute
1. **Fork** the repository  
2. **Create a feature branch**  
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. Make your changes and commit
```bash
git commit -m "Add: YourFeatureName"
```
4. Push to your fork
```bash
git push origin feature/YourFeatureName
```
5. Open a Pull Request on GitHub


