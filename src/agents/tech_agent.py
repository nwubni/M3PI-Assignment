"""
Tech Agent class that processes user queries related to Technical Support.
"""

from typing import Optional
from agents.agent import Agent


class TechAgent(Agent):
    """
    Tech Agent class that processes user queries related to Technical Support.
    """

    def __init__(self):
        super().__init__("TechAgent")
