"""
Multi-agent system to classify and route user queries to appropriate agents.
"""

from dotenv import load_dotenv
from langfuse import Langfuse

from agents.orchestrator import Orchestrator

load_dotenv()


def main():
    """
    Main function to run the orchestrator with automatic evaluation.
    """

    orchestrator = Orchestrator()

    while True:
        query = input("What would you like to know?: ")
        orchestrator.set_query(query)

        if query.lower() in ["exit", "quit", "q"]:
            break

        response = orchestrator.run()
        print(f"\n{'='*60}")
        print("FINAL RESPONSE")
        print(f"{'='*60}")
        print(f"{response}")
        print(f"{'='*60}\n")

    print("Flushing Langfuse traces...")
    Langfuse().flush()
    print("Program Exited Successfully")


if __name__ == "__main__":
    main()
