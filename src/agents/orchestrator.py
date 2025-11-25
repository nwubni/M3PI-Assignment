"""
Orchestrator class that classifies user queries into agent categories.
"""

import os
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from agents.finance_agent import FinanceAgent
from agents.hr_agent import HRAgent
from agents.tech_agent import TechAgent

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


load_dotenv()

# Agent mapping for tool creation
agents = {"HRAgent": HRAgent, "TechAgent": TechAgent, "FinanceAgent": FinanceAgent}


def create_agent_qa_chain(agent: str, query: str):
    """
    Creates and runs a specific agent with the given query using LangChain 1.x Runnable pipeline.

    Args:
        agent (str): The name of the agent.
        query (str): The query to be processed.

    Returns:
        str: The agent's response based on retrieved context.
    """
    agent_obj = agents[agent]()
    agent_obj.query = query

    try:
        # Get the vector store and create retriever
        vector_store = agent_obj.get_local_index()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Create LLM
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            temperature=0,
            max_completion_tokens=130,
        )

        # Load prompt template from file
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "agent_prompt.txt")
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create the Runnable pipeline using LangChain 1.x API
        qa_chain = (
            RunnableMap(
                {
                    "context": lambda x: "\n\n".join(
                        [
                            doc.page_content
                            for doc in retriever.invoke(x["question"])
                        ]
                    ),
                    "question": lambda x: x["question"],
                    "agent_type": lambda x: x["agent_type"],
                }
            )
            | prompt
            | llm
        )

        # Run the chain
        result = qa_chain.invoke(
            {"question": query, "agent_type": agent.replace("Agent", "")}
        )

        return result.content

    except Exception as e:
        # Fallback to basic agent response if vector store fails
        print(f"Vector store error for {agent}: {e}")
        return agent_obj.run()


# Create wrapper functions for each agent
def hr_agent_func(query: str) -> str:
    """Wrapper function for HRAgent"""
    return create_agent_qa_chain("HRAgent", query)


def tech_agent_func(query: str) -> str:
    """Wrapper function for TechAgent"""
    return create_agent_qa_chain("TechAgent", query)


def finance_agent_func(query: str) -> str:
    """Wrapper function for FinanceAgent"""
    return create_agent_qa_chain("FinanceAgent", query)


tools = [
    Tool(
        name="HRAgent",
        func=hr_agent_func,
        description="Answers questions related to Human Resources (policies, benefits, etc).",
    ),
    Tool(
        name="TechAgent",
        func=tech_agent_func,
        description="Answers Technical Support questions (IT issues, software, etc).",
    ),
    Tool(
        name="FinanceAgent",
        func=finance_agent_func,
        description="Answers questions related to Finance (budget, expenses, etc).",
    ),
]


class Orchestrator:
    """
    Orchestrator class that classifies user queries into agent categories.
    """

    def __init__(self, query: str):
        self.query = query
        prompt_text = open("prompts/orchestrator_prompt.txt", "r").read()

        prompt_text = prompt_text.replace("User query: {query}", "").strip()
        self.system_prompt = prompt_text

        model_name = os.getenv("LLM_MODEL", "gpt-4")
        self.model_string = f"openai:{model_name}"

        print(f"Orchestrator is initializing with query: {self.query}")

    def run(self) -> str:
        """
        Runs the orchestrator with tool-based agent routing.

        Returns:
            str: The agent's response to the query.
        """

        model = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4"), temperature=0)

        # Create the agent with tools for routing
        agent_graph = create_agent(
            model=model, tools=tools, system_prompt=self.system_prompt
        )

        # Invoke the agent with messages
        result = agent_graph.invoke({"messages": [HumanMessage(content=self.query)]})

        # Extract the last AI message content from the result
        if result.get("messages"):
            last_message = result["messages"][-1]
            return (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )

        return str(result)
