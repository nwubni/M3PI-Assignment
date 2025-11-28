"""
Quick validation script - tests a few key queries to ensure system is working correctly.
This is used for rapid testing during development.
"""

from pathlib import Path
import sys

from src.agents.orchestrator import Orchestrator

sys.path.append(str(Path(__file__).parent.parent))


def quick_test():
    """Run quick validation on 5 representative queries."""

    test_queries = [
        {"query": "How do I apply for parental leave?", "expected_agent": "HRAgent"},
        {"query": "My laptop won't turn on", "expected_agent": "TechAgent"},
        {
            "query": "How do I submit an expense report?",
            "expected_agent": "FinanceAgent",
        },
        {"query": "What's the company's vacation policy?", "expected_agent": "HRAgent"},
        {"query": "I can't access the VPN", "expected_agent": "TechAgent"},
    ]

    print("\n" + "=" * 60)
    print("QUICK VALIDATION TEST")
    print("=" * 60 + "\n")

    passed = 0

    for i, test in enumerate(test_queries, 1):
        print(f"Test {i}/5: {test['query'][:50]}...")

        try:
            orchestrator = Orchestrator(query=test["query"])
            response = orchestrator.run()

            # Simple check: response should not be empty
            if response and len(response) > 20:
                print(f"  ✓ Response generated ({len(response)} chars)")
                passed += 1
            else:
                print(f"  ✗ Response too short or empty")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    print("=" * 60)
    print(f"Result: {passed}/5 tests passed")
    print("=" * 60 + "\n")

    return passed == 5


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
