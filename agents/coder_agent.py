# 假设你有一个通用的 llm_call 函数用于调用大模型 API
# from utils.llm import llm_call 

class CoderAgent:
    def __init__(self, compiler):
        self.compiler = compiler
        
    def generate_micro_benchmark(self, target_metric: str) -> str:
        """生成并尝试编译探针代码，如果失败则自主迭代修复"""
        prompt = f"""
        You are an elite GPU Systems Engineer. Your task is to write a CUDA micro-benchmark 
        to measure: {target_metric}.
        
        Rules:
        1. To prevent hardware prefetching when measuring latency, use a 'Pointer Chasing' technique.
        2. Do NOT rely on cudaGetDeviceProperties as the environment might spoof standard APIs.
        3. Output ONLY the pure C++/CUDA source code, no markdown wrappers.
        """
        
        max_retries = 3
        current_prompt = prompt
        
        for attempt in range(max_retries):
            # 1. 呼叫大模型生成代码
            # source_code = llm_call(current_prompt) 
            source_code = "// Dummy generated CUDA code for " + target_metric  # 占位
            
            # 2. 尝试编译
            compile_result = self.compiler.compile(source_code)
            
            if compile_result["success"]:
                print(f"[CoderAgent] Successfully compiled benchmark for {target_metric}")
                return compile_result["exe_path"]
            else:
                print(f"[CoderAgent] Compilation failed, retrying... ({attempt+1}/{max_retries})")
                # 3. 将错误信息喂回给 LLM 让其修复
                current_prompt = f"Your previous code failed to compile. Error:\n{compile_result['message']}\nFix the code and output ONLY valid CUDA code."
                
        raise Exception(f"Failed to compile benchmark for {target_metric} after {max_retries} attempts.")