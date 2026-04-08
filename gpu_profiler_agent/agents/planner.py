from core.base_agent import BaseAgent
from core.state import WorkflowState
import asyncio

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("PlannerAgent")

    async def run(self, state: WorkflowState) -> WorkflowState:
        self.logger.info("PlannerAgent started")
        state.add_reasoning("[Planner] Analyzing targets from target_spec.json...")
        
        for target in state.targets:
            target_type = target.get("type", "mock_target")
            target_name = target.get("name", "unknown")
            state.add_reasoning(f"[Planner] Target '{target_name}' mapped to probe strategy '{target_type}'")
            
        await asyncio.sleep(0.5)  # Simulate planning time
        return state
