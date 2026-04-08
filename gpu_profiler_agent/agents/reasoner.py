from core.base_agent import BaseAgent
from core.state import WorkflowState
from typing import Dict, Any, List, Tuple

class NCUAnalyzer:
    """
    Implements a strict 4-Step NCU diagnostic logic based on Roofline and pipeline analysis.
    """
    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics
        self.diagnostics: List[str] = []
        self.suggestions: List[str] = []

    def get_metric(self, key: str, default: float = 0.0) -> float:
        return float(self.metrics.get(key, default))

    def analyze(self) -> Tuple[List[str], List[str]]:
        # --- Step 1: Roofline Direction ---
        sm_throughput = self.get_metric("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        mem_throughput = self.get_metric("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
        
        self.diagnostics.append(f"[Step 1] Roofline: SM={sm_throughput:.1f}%, Mem={mem_throughput:.1f}%")

        # --- Step 2: Characterize ---
        if mem_throughput > sm_throughput:
            dram_throughput = self.get_metric("dram__throughput.avg.pct_of_peak_sustained_elapsed")
            if dram_throughput > 70.0:
                self.diagnostics.append("[Step 2] Memory > Compute & DRAM > 70%. Diagnosed as [VRAM Bound].")
                self.suggestions.append("VRAM Bound: Increase data reuse in Shared Memory or Registers to reduce global memory traffic.")
            else:
                self.diagnostics.append("[Step 2] Memory > Compute, but DRAM is not saturated. Possibly latency bound or L2 bound.")
        elif sm_throughput >= mem_throughput and sm_throughput > 0:
            tensor_active = self.get_metric("sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active")
            if tensor_active > 0:
                self.diagnostics.append(f"[Step 2] Compute > Memory. Tensor Cores are active ({tensor_active:.1f}%).")
            else:
                self.diagnostics.append("[Step 2] Compute > Memory, but Tensor Cores NOT utilized.")
                self.suggestions.append("Compute Bound: Refactor kernel to use HMMA instructions (FP16/TF32) to leverage Tensor Cores.")

        # --- Step 3: Anomalies ---
        # 1. Uncoalesced Access
        uncoalesced = self.get_metric("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        if uncoalesced > 10000: # Threshold is context-dependent, using 10000 as a heuristical mock
            self.diagnostics.append(f"[Step 3] High global memory sectors loaded: {uncoalesced}.")
            self.suggestions.append("Uncoalesced Access: Ensure threads in a warp access contiguous memory addresses (stride-1).")

        # 2. Warp Divergence
        warp_efficiency = self.get_metric("sm__sass_thread_inst_executed_per_inst_executed", 32.0)
        if warp_efficiency < 32.0:
            self.diagnostics.append(f"[Step 3] Warp divergence detected. Efficiency: {warp_efficiency:.1f}/32.")
            self.suggestions.append("Warp Divergence: Reduce if/else branching or group threads by execution path.")

        # 3. Occupancy Anomaly
        theo_occ = self.get_metric("sm__maximum_warps_per_active_cycle_pct", 100.0)
        achieved_occ = self.get_metric("sm__warps_active.avg.pct_of_peak_sustained_active", theo_occ)
        if (theo_occ - achieved_occ) > 20.0:
            self.diagnostics.append(f"[Step 3] Occupancy gap: Theoretical={theo_occ:.1f}%, Achieved={achieved_occ:.1f}%")
            self.suggestions.append("Occupancy Gap: Check for register pressure, shared memory limits, or unbalanced block scheduling.")

        # 4. Bank Conflicts
        bank_conflicts = self.get_metric("l1tex__data_bank_conflicts_pipe_lsu.sum")
        if bank_conflicts > 0:
            self.diagnostics.append(f"[Step 3] Detected {int(bank_conflicts)} shared memory bank conflicts.")
            self.suggestions.append("Bank Conflicts: Pad shared memory arrays or restructure access patterns to avoid strided accesses.")

        return self.diagnostics, self.suggestions

class ReasonerAgent(BaseAgent):
    def __init__(self):
        super().__init__("ReasonerAgent")

    async def run(self, state: WorkflowState) -> WorkflowState:
        self.logger.info("ReasonerAgent analyzing extracted metrics...")
        state.add_reasoning("[Reasoner] Starting NCU Analyzer engine...")
        
        for target_name, metrics in state.extracted_metrics.items():
            state.add_reasoning(f"\n[Reasoner] --- Analyzing Target: {target_name} ---")
            
            analyzer = NCUAnalyzer(metrics)
            diagnostics, suggestions = analyzer.analyze()
            
            for diag in diagnostics:
                state.add_reasoning(f"  {diag}")
            
            if suggestions:
                state.add_reasoning("  [Step 4] Optimization Suggestions:")
                for sugg in suggestions:
                    state.add_reasoning(f"    -> {sugg}")
            else:
                state.add_reasoning("  [Step 4] Optimization Suggestions: None. Code looks optimal!")
                
        # Handle execution failures via stderr loop (Mock implementation)
        for result in state.execution_results:
            if not result.success and result.stderr:
                state.add_reasoning(f"[Reasoner] Error detected: {result.stderr.strip()}. Emitting repair suggestions...")
                
        return state
