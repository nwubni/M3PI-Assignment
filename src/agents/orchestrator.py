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
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

# Add path for evaluator import
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.enums.agent_enums import AgentType
from src.agents.finance_agent import FinanceAgent
from src.agents.hr_agent import HRAgent
from src.agents.tech_agent import TechAgent

load_dotenv()

# Import evaluator
try:
    from evaluator.evaluator import evaluate_rag_quality

    EVALUATOR_AVAILABLE = True
except ImportError:
    print("Warning: Evaluator not available")
    EVALUATOR_AVAILABLE = False

# Agent mapping for tool creation
agents = {
    AgentType.HR.value: HRAgent,
    AgentType.TECH.value: TechAgent,
    AgentType.FINANCE.value: FinanceAgent,
}


@observe(name="create_agent_qa_chain", capture_input=True, capture_output=True)
def create_agent_qa_chain(agent: str, query: str, langfuse_handler=None):
    """
    Creates and runs a specific agent with the given query using LangChain 1.x Runnable pipeline.

    Args:
        agent (str): The name of the agent.
        query (str): The query to be processed.
        langfuse_handler: LangFuse callback handler for tracing.

    Returns:
        tuple: (response_text, retrieved_chunks_data) for logging to LangFuse
    """
    agent_obj = agents[agent]()
    model_name = os.getenv("LLM_MODEL", "gpt-4")

    try:
        # Get the vector store and create retriever
        vector_store = agent_obj.get_local_index()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Retrieve documents
        retrieved_docs = retriever.invoke(query)

        # Log retrieved chunks to console and prepare for LangFuse
        print(f"\nRetrieved {len(retrieved_docs)} chunks for {agent}:")
        chunks_data = []
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"  Chunk {i}: {doc.page_content[:100]}...")
            chunks_data.append(
                {"chunk_id": i, "content": doc.page_content, "metadata": doc.metadata}
            )
        print()

        # Create LLM with callback handler
        # CallbackHandler automatically tracks: tokens, costs, latency, model, input, output
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_completion_tokens=130,
            callbacks=[langfuse_handler] if langfuse_handler else [],
        )

        # Load prompt template from file
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "prompts", "agent_prompt.txt"
        )
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create the Runnable pipeline using LangChain 1.x API
        qa_chain = (
            RunnableMap(
                {
                    "context": lambda x: "\n\n".join(
                        [doc.page_content for doc in retrieved_docs]
                    ),
                    "question": lambda x: x["question"],
                    "agent_type": lambda x: x["agent_type"],
                }
            )
            | prompt
            | llm
        )

        # Run the chain with callback
        # CallbackHandler will automatically track tokens, costs, latency, model, input, output
        config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
        result = qa_chain.invoke(
            {"question": query, "agent_type": agent.replace("Agent", "")}, config=config
        )

        # Return response and chunks data
        # The @observe decorator will capture this return value
        # Chunks are included in the output for LangFuse tracking
        return {
            "response": result.content,
            "chunks": chunks_data,
            "num_chunks": len(chunks_data),
            "agent": agent,
            "model": model_name,
        }

    except Exception as e:
        # Fallback to basic agent response if vector store fails
        print(f"Vector store error for {agent}: {e}")
        # Error is automatically captured by @observe decorator
        # Return in consistent format
        fallback_response = agent_obj.run()
        return {
            "response": fallback_response,
            "chunks": [],
            "num_chunks": 0,
            "agent": agent,
            "model": model_name,
            "error": str(e),
        }


# Create wrapper functions for each agent
# @observe decorator automatically captures: input, output, latency
# CallbackHandler (passed to create_agent_qa_chain) automatically captures: tokens, costs, model
@observe(name="hr_agent", capture_input=True, capture_output=True)
def hr_agent_func(query: str) -> str:
    """Wrapper function for HRAgent"""
    # Create CallbackHandler - it will automatically track tokens, costs, latency, model
    # The handler is part of the trace created by @observe decorator
    langfuse_handler = CallbackHandler()
    result = create_agent_qa_chain("HRAgent", query, langfuse_handler)

    # Handle dict return (response, chunks_data, metadata)
    if isinstance(result, dict):
        return result.get("response", str(result))
    # Handle tuple return for backward compatibility
    if isinstance(result, tuple):
        return result[0]
    return result


@observe(name="tech_agent", capture_input=True, capture_output=True)
def tech_agent_func(query: str) -> str:
    """Wrapper function for TechAgent"""
    # Create CallbackHandler - it will automatically track tokens, costs, latency, model
    langfuse_handler = CallbackHandler()
    result = create_agent_qa_chain("TechAgent", query, langfuse_handler)

    # Handle dict return (response, chunks_data, metadata)
    if isinstance(result, dict):
        return result.get("response", str(result))
    # Handle tuple return for backward compatibility
    if isinstance(result, tuple):
        return result[0]
    return result


@observe(name="finance_agent", capture_input=True, capture_output=True)
def finance_agent_func(query: str) -> str:
    """Wrapper function for FinanceAgent"""
    # Create CallbackHandler - it will automatically track tokens, costs, latency, model
    langfuse_handler = CallbackHandler()
    result = create_agent_qa_chain("FinanceAgent", query, langfuse_handler)

    # Handle dict return (response, chunks_data, metadata)
    if isinstance(result, dict):
        return result.get("response", str(result))
    # Handle tuple return for backward compatibility
    if isinstance(result, tuple):
        return result[0]
    return result


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

    @observe(name="orchestrator", capture_input=True, capture_output=True)
    def run(self) -> str:
        """
        Runs the orchestrator with tool-based agent routing.

        LangFuse automatically tracks:
        - Input/Output (via @observe decorator)
        - Latency (via @observe decorator)
        - Tokens, Costs, Model (via CallbackHandler on LLM calls)
        - Chunks (via metadata in create_agent_qa_chain)

        Returns:
            str: The agent's response to the query.
        """
        model_name = os.getenv("LLM_MODEL", "gpt-4")

        # Create LangFuse callback handler for the orchestrator
        # CallbackHandler automatically tracks: tokens, costs, latency, model, input, output
        langfuse_handler = CallbackHandler()

        # Add metadata using a span (metadata is automatically captured by @observe decorator)
        # The @observe decorator already captures input/output, and CallbackHandler captures model/tokens/costs

        model = ChatOpenAI(
            model=model_name,
            temperature=0,
            callbacks=[
                langfuse_handler
            ],  # Automatically tracks tokens, costs, latency, model
        )

        # Create the agent with tools for routing
        agent_graph = create_agent(
            model=model, tools=tools, system_prompt=self.system_prompt
        )

        # Invoke the agent with messages and callback
        # CallbackHandler will track all LLM calls within the agent execution
        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=self.query)]},
            config={"callbacks": [langfuse_handler]},
        )

        # Extract the last AI message content from the result
        if result.get("messages"):
            last_message = result["messages"][-1]
            response = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )
        else:
            response = str(result)

        # Automatically evaluate the response
        self._evaluate_response(response)

        return response

    def _evaluate_response(self, response: str):
        """
        Automatically evaluates the response quality and logs score to Langfuse.

        Args:
            response: The agent's response to evaluate
        """
        if not EVALUATOR_AVAILABLE:
            return

        try:
            # Evaluate the response
            eval_result = evaluate_rag_quality(
                query=self.query,
                response=response,
                context=None,  # Context is already in the prompt
            )

            # Use the Langfuse client to score the current trace
            # This works because we're inside an @observe decorated function
            langfuse = get_client()

            # Round score to whole number and score the current trace
            rounded_score = round(eval_result["score"])

            langfuse.score_current_trace(
                name="Evaluation Score (0-10)",
                value=rounded_score,
                data_type="NUMERIC",
                comment=eval_result["reasoning"],
            )

            print(f"✅ Score logged to Langfuse trace: {rounded_score}/10")

            # Print evaluation results
            print(f"\n{'='*60}")
            print("📊 QUALITY EVALUATION")
            print(f"{'='*60}")
            print(f"Score: {eval_result['score']}/10")
            print(f"Reasoning: {eval_result['reasoning']}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"⚠️  Evaluation failed: {e}")
            import traceback

            traceback.print_exc()
