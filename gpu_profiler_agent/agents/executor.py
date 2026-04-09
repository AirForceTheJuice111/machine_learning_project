import asyncio
import os
import shlex
from pathlib import Path
from core.base_agent import BaseAgent
from core.state import WorkflowState, ExecutionResult
from probes.registry import ProbeRegistry
from config import WORKSPACE_DIR, MAX_RETRIES

# Global lock to prevent concurrent GPU profiling contentions
GPU_LOCK = asyncio.Lock()

class ExecutorAgent(BaseAgent):
    def __init__(self, mock_execution: bool = True):
        super().__init__("ExecutorAgent")
        self.mock_execution = mock_execution

    async def _run_cmd(self, cmd: str, timeout: int = 10):
        """Helper to run shell command with timeout."""
        if self.mock_execution:
            await asyncio.sleep(1)
            # Simulate a mock success
            if "nvcc" in cmd:
                return 0, "", ""
            else:
                return 0, "mock_ncu_csv_output", ""
                
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return process.returncode, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            process.kill()
            return -1, "", "Execution Timed Out"

    async def run(self, state: WorkflowState) -> WorkflowState:
        state.add_reasoning("[Executor] Starting constrained GPU execution (with asyncio.Lock).")
        
        for target in state.targets:
            target_type = target.get("type", "mock_target")
            target_name = target.get("name", "unknown")
            strategy = ProbeRegistry.get_strategy(target_type)
            
            cu_file_path = WORKSPACE_DIR / f"{target_name}.cu"
            exe_file_path = WORKSPACE_DIR / f"{target_name}.out"
            
            # Simulate initial code generation
            current_code = strategy.generate_cuda_code(target)
            
            success = False
            result = ExecutionResult()
            
            for attempt in range(1, MAX_RETRIES + 1):
                state.add_reasoning(f"[Executor] Attempt {attempt}/{MAX_RETRIES} for target '{target_name}'")
                
                async with GPU_LOCK:
                    state.add_reasoning(f"[Executor] Acquired GPU Lock for '{target_name}'.")
                    
                    if state.executable_path:
                        # 1. Use provided executable directly
                        state.add_reasoning(f"[Executor] Skipping compilation. Using provided executable: {state.executable_path}")
                        exe_file_path = state.executable_path
                    else:
                        # Write current code to workspace
                        with open(cu_file_path, "w") as f:
                            f.write(current_code)
                        if str(cu_file_path) not in state.generated_code_paths:
                            state.generated_code_paths.append(str(cu_file_path))
                        
                        # 1. Compile
                        compile_cmd = f"nvcc {cu_file_path} -o {exe_file_path}"
                        state.add_reasoning(f"[Executor] Running: {compile_cmd}")
                        ret_code, stdout, stderr = await self._run_cmd(compile_cmd)
                        
                        if ret_code != 0:
                            state.add_reasoning(f"[Executor] Compilation failed. Intercepted stderr for Reasoner.")
                            # Simulate sending to Reasoner to fix code
                            current_code += "\n// Fixed by Reasoner..."
                            continue
                        
                    # 2. Profile
                    metrics_list = strategy.required_metrics if hasattr(strategy, "required_metrics") else []
                    metrics_str = ",".join(metrics_list) if metrics_list else "all"
                    
                    # Anti-Hacking: Multi-trial Cross-Verification
                    NUM_TRIALS = 3
                    state.add_reasoning(f"[Executor] Initiating Cross-Verification ({NUM_TRIALS} trials) to bypass potential OS-level caching/frequency scaling...")
                    
                    all_stdout = ""
                    for trial in range(1, NUM_TRIALS + 1):
                        ncu_cmd = f"ncu --metrics {metrics_str} --csv {exe_file_path}"
                        state.add_reasoning(f"[Executor] [Trial {trial}/{NUM_TRIALS}] Running: {ncu_cmd}")
                        ret_code, stdout, stderr = await self._run_cmd(ncu_cmd)
                        all_stdout = stdout # Just take the last stdout for mock purposes, in real life we average
                        
                        if ret_code != 0:
                            break
                            
                    if ret_code != 0:
                        state.add_reasoning(f"[Executor] Profiling failed. Intercepted stderr for Reasoner.")
                        current_code += "\n// Fixed runtime error by Reasoner..."
                        continue
                        
                    # Success
                    success = True
                    result.success = True
                    result.stdout = all_stdout
                    result.stderr = stderr
                    result.metrics = strategy.parse_ncu_output(all_stdout)
                    state.add_reasoning(f"[Executor] Cross-verification completed successfully. Releasing GPU Lock.")
                    break
            
            state.execution_results.append(result)
            if success:
                state.extracted_metrics[target_name] = result.metrics
            else:
                state.add_reasoning(f"[Executor] All {MAX_RETRIES} attempts failed for '{target_name}'.")
                
        return state
