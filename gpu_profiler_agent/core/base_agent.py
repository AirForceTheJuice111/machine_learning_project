from abc import ABC, abstractmethod
from core.state import WorkflowState
import logging

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(self.name)
        
    @abstractmethod
    async def run(self, state: WorkflowState) -> WorkflowState:
        """Execute agent logic and update state."""
        pass
