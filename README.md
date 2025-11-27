# M3PI-Assignment
Andela GenAI Third Assignment - Multi-Agent RAG System with Automated Quality Evaluation

## Overview

A production-ready multi-agent orchestration system that routes user queries to specialized RAG agents and automatically evaluates response quality using LLM-as-judge.

## Features

✅ **Multi-Agent Orchestration** - Intent classification and conditional routing  
✅ **3 Specialized RAG Agents** - HR, Tech Support, and Finance domains  
✅ **LangChain Integration** - Production-grade components (chains, retrievers, agents)  
✅ **Complete Langfuse Tracing** - Full workflow observability and debugging  
✅ **Automated Quality Evaluation** - Every response scored 1-10 with reasoning  
✅ **Domain-Specific Knowledge** - Grounded in company documentation  

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the system
python src/multi_agent_system.py

# Test evaluator integration
python test_evaluator.py
```

## Architecture

```
User Query
    ↓
Orchestrator (Intent Classification)
    ↓
[HRAgent | TechAgent | FinanceAgent]
    ↓
RAG Pipeline (Retrieval + Generation)
    ↓
Automated Evaluator (Quality Scoring)
    ↓
Response + Score logged to Langfuse
```

## Key Components

### 1. Orchestrator (`src/agents/orchestrator.py`)
- Classifies user intent using LLM
- Routes to appropriate specialized agent
- Automatically evaluates responses

### 2. Specialized Agents
- **HRAgent**: Employee policies, benefits, HR procedures
- **TechAgent**: IT support, troubleshooting, technical issues
- **FinanceAgent**: Budget, expenses, financial processes

### 3. Automated Evaluator (`evaluator/evaluator.py`)
- LLM-as-judge scoring (1-10 scale)
- Evaluates: relevance, accuracy, completeness, clarity
- Logs scores to Langfuse for monitoring

### 4. Langfuse Integration
- Complete trace capture
- Token/cost tracking
- Quality score logging
- Production debugging

## Documentation

- **[Evaluator Integration](EVALUATOR_INTEGRATION.md)** - How automated evaluation works
- **[Technical Overview](reports/overview_report.md)** - System architecture and design

## What Gets Tracked in Langfuse

Every query generates a trace with:
- Input/output for each step
- Retrieved document chunks
- Token counts and costs
- Model information
- Latency metrics
- **Quality score (1-10) with reasoning**

## Testing

```bash
# Run full system test
python test_evaluator.py

# Run single query
python src/multi_agent_system.py
```

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=your-key
LLM_MODEL=gpt-4o-mini
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Project Structure

```
M3PI-Assignment/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py      # Main routing logic
│   │   ├── hr_agent.py          # HR specialist
│   │   ├── tech_agent.py        # Tech support specialist
│   │   └── finance_agent.py     # Finance specialist
│   └── multi_agent_system.py    # Entry point
├── evaluator/
│   └── evaluator.py             # Automated quality scoring
├── data/
│   ├── hr_docs/                 # HR knowledge base
│   ├── tech_docs/               # Tech knowledge base
│   └── finance_docs/            # Finance knowledge base
├── prompts/
│   ├── orchestrator_prompt.txt  # Routing instructions
│   └── agent_prompt.txt         # RAG response template
└── storage/                     # Vector store indices
```

## Monitoring Quality

In Langfuse dashboard, track:
1. Average quality score over time
2. Low-scoring responses (< 6/10)
3. Score distribution by agent
4. Correlation with latency/cost

## Next Steps

- Set up Langfuse alerts for low scores
- A/B test different prompts using scores
- Implement score-based retry logic
- Add custom evaluation criteria
