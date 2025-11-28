"""
Multi-agent system to classify and route user queries to appropriate agents.
"""

from dotenv import load_dotenv
from langfuse import Langfuse

from agents.orchestrator import Orchestrator

# Load environment variables
load_dotenv()


def main():
    """
    Main function to run the orchestrator with automatic evaluation.

    LangFuse will automatically track:
    - Input/Output (via @observe decorators)
    - Latency (via @observe decorators)
    - Tokens (via CallbackHandler on LLM calls)
    - Costs (via CallbackHandler on LLM calls)
    - Model (via CallbackHandler on LLM calls)
    - Chunks (via metadata in create_agent_qa_chain)
    - Quality Scores (via automated evaluator)
    """

    while True:
        query = input("What would you like to know?: ")

        if query.lower() in ["exit", "quit", "q"]:
            break

        orchestrator = Orchestrator(query=query)
        response = orchestrator.run()
        print(f"\n{'='*60}")
        print("FINAL RESPONSE")
        print(f"{'='*60}")
        print(f"{response}")
        print(f"{'='*60}\n")

    # Flush LangFuse to ensure all async traces and scores are sent before program exits

    print("Flushing Langfuse traces...")
    Langfuse().flush()
    print("✓ All traces and scores sent to Langfuse dashboard")


if __name__ == "__main__":
    main()
