# Reflexion Agent

An AI-powered research assistant implementing the Reflexion pattern - an iterative self-reflection and improvement loop using LangGraph.

## What is Reflexion?

Reflexion is an agent architecture that learns from mistakes through self-reflection:

1. **Actor** - Generates initial responses to research questions
2. **Evaluator** - Uses search tools to validate and gather evidence
3. **Self-Reflection** - Critiques the response and identifies gaps
4. **Memory** - Stores reflections to inform subsequent iterations
5. **Iterative Refinement** - Revises answers based on critique and new information

## Architecture

```
User Question
     ↓
[Draft Answer] → [Execute Search] → [Revise Answer]
                         ↑                ↓
                         └────────────────┘
                    (Iterate until max revisions)
```

### Components

- **[schemas.py](schemas.py)** - Pydantic models for structured outputs (`AnswerQuestion`, `ReviseAnswer`)
- **[chains.py](chains.py)** - LangChain prompt templates and LLM chains (Actor & Revisor)
- **[tool_executor.py](tool_executor.py)** - Search functionality using Tavily API
- **[graph.py](graph.py)** - LangGraph workflow with state management and nodes
- **[main.py](main.py)** - Entry point to run the agent

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

3. Get API keys:
   - OpenAI API key from https://platform.openai.com/
   - Tavily API key from https://tavily.com/

## Usage

Run the agent:
```bash
python main.py
```

To customize the question, edit the `question` variable in [main.py](main.py:8).

## How It Works

1. **Draft Phase**: Actor generates an initial answer and search queries
2. **Search Phase**: Executes searches to gather supporting evidence
3. **Revision Phase**: Revisor critiques the answer and generates an improved version
4. **Loop**: Process repeats until max revisions reached or no more queries

## Configuration

Adjust these parameters in [main.py](main.py:23):
- `max_revisions`: Maximum number of revision iterations (default: 2)
- `question`: The research question to answer

Model settings in [chains.py](chains.py:7):
- `model`: GPT model to use (default: "gpt-4o-mini")
- `temperature`: Creativity level (default: 0)

Search settings in [tool_executor.py](tool_executor.py:9):
- `max_results`: Number of search results per query (default: 3)
