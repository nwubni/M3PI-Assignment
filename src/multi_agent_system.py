"""
Multi-agent system to classify and route user queries to appropriate agents.
"""

from agents.orchestrator import Orchestrator


def main():
    """
    Main function to run the orchestrator.
    """
    orchestrator = Orchestrator(query="Can I work remotely, and what does it take?")
    response = orchestrator.run()
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
