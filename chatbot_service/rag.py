from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import HuggingFaceHub
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
import os

# Optional: Set HuggingFace API token (free account needed)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# Free web search
search = DuckDuckGoSearchRun()

# Free HuggingFace model (like bigscience/bloom or google/flan-t5)
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 512})

tools = [
    Tool(
        name="Web Search",
        func=search.run,
        description="Useful for answering questions about current events or factual topics."
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def ask_with_web(query):
    return agent.run(query)
