"""
Tech Agent class that processes user queries related to Technical Support.
"""

from src.enums.agent_enums import AgentType
from src.agents.agent import Agent


class TechAgent(Agent):
    """
    Tech Agent class that processes user queries related to Technical Support.
    """

    def __init__(self):
        super().__init__(AgentType.TECH)
