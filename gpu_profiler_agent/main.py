import asyncio
import json
import os
from pathlib import Path
from rich.live import Live

from config import RESULTS_FILE, REASONING_LOG_FILE
from core.state import WorkflowState
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.reasoner import ReasonerAgent
from utils.ui_renderer import UIRenderer
import probes.hardware_probes  # Import to register hardware strategies

async def run_agents(state: WorkflowState, ui: UIRenderer):
    # Planner phase
    ui.update_executor_status("PlannerAgent Running...")
    planner = PlannerAgent()
    state = await planner.run(state)
    
    # Executor phase
    ui.update_executor_status("ExecutorAgent Compiling & Profiling...")
    executor = ExecutorAgent(mock_execution=True)
    state = await executor.run(state)
    
    # Reasoner phase
    ui.update_executor_status("ReasonerAgent Analyzing Logs...")
    reasoner = ReasonerAgent()
    state = await reasoner.run(state)
    
    ui.update_executor_status("Workflow Completed.")
    return state

async def main():
    state = WorkflowState()
    ui = UIRenderer()
    
    # Read target_spec.json
    spec_path = Path("target_spec.json")
    if spec_path.exists():
        with open(spec_path, "r") as f:
            state.targets = json.load(f)
        state.add_reasoning(f"[System] Loaded {len(state.targets)} targets from target_spec.json")
    else:
        state.add_reasoning(f"[Error] {spec_path} not found.")
        
    # Start UI Live display and async tasks
    async def ui_updater(live: Live):
        while ui.current_executor_status != "Workflow Completed.":
            live.update(ui.render(state))
            await asyncio.sleep(0.1)
        live.update(ui.render(state))  # Final frame
            
    with Live(ui.render(state), refresh_per_second=10) as live:
        ui_task = asyncio.create_task(ui_updater(live))
        state = await run_agents(state, ui)
        await ui_task
        
    # Output results.json
    with open(RESULTS_FILE, "w") as f:
        json.dump(state.extracted_metrics, f, indent=4)
        
    # Output agent_reasoning.log
    with open(REASONING_LOG_FILE, "w") as f:
        f.write("\n".join(state.reasoning_history))
        
    print(f"\n[Success] Profiling complete.")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"Reasoning log saved to {REASONING_LOG_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
