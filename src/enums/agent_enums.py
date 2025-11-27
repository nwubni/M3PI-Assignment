"""
Enum definitions for the multi-agent system.
"""

from enum import Enum


class AgentType(Enum):
    """
    Enum for agent types used in the multi-agent system.
    """

    HR = "HRAgent"
    FINANCE = "FinanceAgent"
    TECH = "TechAgent"
