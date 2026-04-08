import os
from jinja2 import Environment, FileSystemLoader
from probes.registry import ProbeStrategy, ProbeRegistry
from config import TEMPLATES_DIR

def render_template(template_name: str, context: dict) -> str:
    """Helper to render Jinja2 templates for CUDA kernels."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template(template_name)
    return template.render(context)

@ProbeRegistry.register("boost_clock")
class ActualBoostClockProbe(ProbeStrategy):
    def generate_cuda_code(self, target: dict) -> str:
        # Default dense iteration count
        context = {"iterations": target.get("iterations", 1000000)}
        return render_template("boost_clock.cu.j2", context)
        
    def parse_ncu_output(self, stdout: str) -> dict:
        # Mock ncu parsing. In real phase, use regex or csv parser on stdout
        return {
            "sm__sass_thread_inst_executed_op_fp32_pred_on.sum": 1.5e10,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 99.9,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 10.0
        }

@ProbeRegistry.register("memory_latency")
class MemoryLatencyHierarchyProbe(ProbeStrategy):
    def generate_cuda_code(self, target: dict) -> str:
        context = {
            "array_size": target.get("array_size", 1048576), # 1MB
            "stride": target.get("stride", 64)               # Cache line size
        }
        return render_template("memory_latency.cu.j2", context)
        
    def parse_ncu_output(self, stdout: str) -> dict:
        return {
            "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 5000,
            "sm__sass_thread_inst_executed_per_inst_executed": 1.0 # Diverged
        }

@ProbeRegistry.register("effective_bandwidth")
class EffectiveBandwidthProbe(ProbeStrategy):
    def generate_cuda_code(self, target: dict) -> str:
        context = {"array_size": target.get("array_size", 10485760)} # 40MB
        return render_template("effective_bandwidth.cu.j2", context)
        
    def parse_ncu_output(self, stdout: str) -> dict:
        return {
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
            "l1tex__data_bank_conflicts_pipe_lsu.sum": 0,
            "sm__sass_thread_inst_executed_per_inst_executed": 32.0,
            "sm__maximum_warps_per_active_cycle_pct": 100.0,
            "sm__warps_active.avg.pct_of_peak_sustained_active": 95.0
        }

@ProbeRegistry.register("bank_conflict")
class BankConflictPenaltyProbe(ProbeStrategy):
    def generate_cuda_code(self, target: dict) -> str:
        context = {"padding_size": target.get("padding_size", 0)}
        return render_template("bank_conflict.cu.j2", context)
        
    def parse_ncu_output(self, stdout: str) -> dict:
        # Mocking bank conflict data
        padding = 0 # In real scenario this would depend on the stdout or context
        conflicts = 1024 if padding == 0 else 0
        return {
            "l1tex__data_bank_conflicts_pipe_lsu.sum": conflicts,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 20.0
        }
