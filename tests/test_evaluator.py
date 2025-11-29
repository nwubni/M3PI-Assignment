"""
Test script to verify evaluator integration.
Run this to test the automated evaluation system.
"""

from pathlib import Path
import sys

# Add parent directory to path so we can import src and evaluator modules
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.agents.orchestrator import Orchestrator
from langfuse import Langfuse

load_dotenv()


def test_evaluator_integration():
    """Test that evaluator runs automatically and logs to Langfuse."""

    print("\n" + "=" * 70)
    print("TESTING AUTOMATED EVALUATOR INTEGRATION")
    print("=" * 70 + "\n")

    # Test queries for different agents
    test_queries = [
        "How do I contact HR?",
        "My laptop is running slow, what should I do?",
        "What is the budget approval process?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*70}")
        print(f"TEST {i}/3: {query}")
        print(f"{'─'*70}\n")

        try:
            # Run orchestrator
            orchestrator = Orchestrator()
            orchestrator.set_query(query)
            response = orchestrator.run()

            print(f"\n✓ Test {i} completed successfully")
            print(f"  Response length: {len(response)} characters")

        except Exception as e:
            print(f"\n✗ Test {i} failed: {e}")

    # Flush Langfuse
    print(f"\n{'='*70}")
    print("Flushing Langfuse traces and scores...")
    Langfuse().flush()
    print("✓ All traces and scores sent to Langfuse")
    print("=" * 70 + "\n")

    print("🎉 Evaluator integration test complete!")
    print("\nCheck your Langfuse dashboard to see:")
    print("  • 3 traces (one for each query)")
    print("  • 3 quality scores (rag_quality_score)")
    print("  • Score values (1-10) with reasoning")
    print("  • Full execution traces with chunks and tokens")


if __name__ == "__main__":
    test_evaluator_integration()
