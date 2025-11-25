"""
Agent class for all agents.
"""

import os

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


class Agent:
    """
    Base class for all agents.
    """

    def __init__(self, name: str):
        self.name = name
        print(f"{self.name} is initializing")

    def get_local_index(self):
        """
        Returns the local index for the agent.
        """

        index_dir = f"storage/vectors/{self.name.lower()}_index"
        embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
        return FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )

    def run(self):
        """
        Runs the agent.
        """

        print(f"{self.name} is running")
        return "creatively answer query using less than 130 words."
