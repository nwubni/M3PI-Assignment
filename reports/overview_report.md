### Project Overview
This project is a multi-agent RAG system that answers user queries related to HR, Tech, and Finance
Uses LangChain to implement solution for ease of use
Uses orchestrator to route queries base on HR, Tech, and Finance, which fetches chuncks specific to those domains for correct answer context.
Uses openai tooling to invoke the right agent for user query.
Unlike previous projects/Assignments where I had to manually implement log information, this project uses LangFuse for all logging operations and tracing to find and relsove potential errors quickly and easily.