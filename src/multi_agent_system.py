from agents.orchestrator import Orchestrator

def main():
    orchestrator = Orchestrator(query="My salary came short this time. What could be wrong?")
    response = orchestrator.run()
    print(f"Response: {response}")

if __name__ == "__main__":
    main()