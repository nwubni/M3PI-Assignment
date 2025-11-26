"""
Multi-agent system to classify and route user queries to appropriate agents.
"""

from agents.orchestrator import Orchestrator


def main():
    """
    Main function to run the orchestrator.
    """
    orchestrator = Orchestrator(query="Is it possible to get paid in advance?")
    response = orchestrator.run()
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
