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

# Build vector indices from documents
python -m src.build.index data/hr_docs/HRAgent.txt
python -m src.build.index data/tech_docs/TechAgent.txt
python -m src.build.index data/finance_docs/FinanceAgent.txt

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

- **[Technical Overview](reports/overview_report.md)** - System architecture and design
- See `evaluator/evaluator.py` for automated evaluation implementation

## What Gets Tracked in Langfuse

Every query generates a trace with:
- Input/output for each step
- Retrieved document chunks
- Token counts and costs
- Model information
- Latency metrics
- **Quality score (1-10) with reasoning**

## Testing

### Quick Tests
```bash
# Run single query
python src/multi_agent_system.py

# Test evaluator integration
python test_evaluator.py

# Quick validation (5 queries)
python tests/quick_validate.py
```

### Golden Data Validation
```bash
# Full validation suite with golden data
python tests/validate_system.py

# Tests routing accuracy, answer quality, and performance
# See tests/README.md for details
```

**Golden Dataset**: 45+ curated test cases covering HR, Tech, and Finance queries
- Routing accuracy validation
- Answer quality scoring
- Category-wise performance analysis

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=your-key
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

Optional:
```
USE_CUDA=true  # Use GPU for embeddings (if available)
```

## Project Structure

```
M3PI-Assignment/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py      # Main routing logic
│   │   ├── agent.py             # Base agent class
│   │   ├── hr_agent.py          # HR specialist
│   │   ├── tech_agent.py        # Tech support specialist
│   │   └── finance_agent.py    # Finance specialist
│   ├── build/
│   │   └── index.py             # Vector index builder
│   ├── enums/
│   │   └── agent_enums.py       # Agent type enumerations
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
├── storage/
│   └── vectors/                  # Vector store indices
│       ├── hragent_index/        # HR agent FAISS index
│       ├── techagent_index/      # Tech agent FAISS index
│       └── financeagent_index/   # Finance agent FAISS index
└── tests/
    ├── golden_data.json         # Test cases
    ├── quick_validate.py        # Quick validation script
    └── validate_system.py       # Full validation suite
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
