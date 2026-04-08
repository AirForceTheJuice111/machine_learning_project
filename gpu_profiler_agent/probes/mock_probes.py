from probes.registry import ProbeStrategy, ProbeRegistry

@ProbeRegistry.register("mock_target")
class MockProbeStrategy(ProbeStrategy):
    def generate_cuda_code(self, target: dict) -> str:
        kernel_name = target.get("name", "mock_kernel")
        return f"""// Mock target CUDA code for {kernel_name}
#include <iostream>
__global__ void {kernel_name}() {{
    // Mock implementation
}}
int main() {{
    {kernel_name}<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}}
"""
        
    def parse_ncu_output(self, stdout: str) -> dict:
        return {"mock_metric_1": 42.0, "mock_metric_2": 99.9}

@ProbeRegistry.register("gemm")
class GEMMProbeStrategy(ProbeStrategy):
    def generate_cuda_code(self, target: dict) -> str:
        return "// Mock GEMM CUDA Code\nint main() { return 0; }"
        
    def parse_ncu_output(self, stdout: str) -> dict:
        return {"flops": 1.2e13, "memory_bandwidth": 900.0}
