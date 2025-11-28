# src/evaluators/rag_quality_evaluator.py
"""
LangFuse evaluator for RAG response quality scoring.
"""

import os
from openai import OpenAI


def evaluate_rag_quality(query: str, response: str, context: str = None) -> dict:
    """
    Evaluates RAG response quality on a 1-10 scale.

    Args:
        query: Original user query
        response: RAG system's response
        context: Retrieved context (optional)

    Returns:
        dict with score, reasoning, and metadata
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    evaluation_prompt = f"""You are an expert evaluator assessing the quality of RAG (Retrieval-Augmented Generation) responses.

Evaluate the following response on a scale of 1-10 based on these criteria:
- Relevance: Does it answer the query?
- Accuracy: Is the information correct?
- Completeness: Is it comprehensive?
- Clarity: Is it well-structured and clear?
- Context Usage: Does it properly use retrieved information?

Query: {query}

Response: {response}

{f"Retrieved Context: {context[:500]}..." if context else ""}

Provide your evaluation in this exact format:
Score: [1-10]
Reasoning: [Brief explanation of the score]
"""

    completion = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": evaluation_prompt}],
        temperature=0,
        max_tokens=200,
    )

    eval_text = completion.choices[0].message.content

    # Parse score
    score_line = [line for line in eval_text.split("\n") if line.startswith("Score:")]
    score = 5  # default
    reasoning = eval_text

    if score_line:
        try:
            score = int(score_line[0].split(":")[1].strip())
        except:
            pass
        reasoning_lines = [
            line for line in eval_text.split("\n") if line.startswith("Reasoning:")
        ]
        if reasoning_lines:
            reasoning = reasoning_lines[0].split(":", 1)[1].strip()

    return {
        "score": score,
        "reasoning": reasoning,
        "criteria": {
            "relevance": True,
            "accuracy": True,
            "completeness": True,
            "clarity": True,
        },
    }
