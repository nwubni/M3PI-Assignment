"""
HR Agent class that processes user queries related to Human Resources.
"""

from src.enums.agent_enums import AgentType
from src.agents.agent import Agent


class HRAgent(Agent):
    """
    HR Agent class that processes user queries related to Human Resources.
    """

    def __init__(self):
        super().__init__(AgentType.HR)
