"""
Finance Agent class that processes user queries related to Finances.
"""

from src.enums.agent_enums import AgentType
from src.agents.agent import Agent


class FinanceAgent(Agent):
    """
    Finance Agent class that processes user queries related to Finances.
    """

    def __init__(self):
        super().__init__(AgentType.FINANCE)
