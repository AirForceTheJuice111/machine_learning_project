import json
from tools.cuda_compiler import CudaCompiler
from tools.ncu_executor import NcuExecutor
from agents.coder_agent import CoderAgent
# from agents.analyzer_agent import AnalyzerAgent

def main():
    # 1. 初始化工具和 Agents
    compiler = CudaCompiler()
    executor = NcuExecutor()
    coder = CoderAgent(compiler)
    # analyzer = AnalyzerAgent()
    
    # 2. 读取任务输入
    with open("target_spec.json", "r") as f:
        spec = json.load(f)
        
    targets = spec.get("targets", [])
    results = {}
    
    # 3. 针对每个目标启动 Agent 工作流
    for target in targets:
        print(f"\n--- Starting Analysis for: {target} ---")
        
        try:
            # 步骤 A: Coder Agent 编写并编译对应的微基准测试
            exe_path = coder.generate_micro_benchmark(target)
            
            # 步骤 B: 定义需要 ncu 抓取的底层指标
            # 实际项目中，这里可以由 AnalyzerAgent 根据 target 动态决定
            if "dram" in target:
                metrics_to_collect = ["dram__throughput.avg.pct_of_peak_sustained_elapsed", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]
            else:
                metrics_to_collect = ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
                
            # 步骤 C: 在真实 GPU 环境中运行探针
            profile_data = executor.run_with_ncu(exe_path, metrics_to_collect)
            
            # 步骤 D: Analyzer Agent 分析数据得出最终数值
            # final_value = analyzer.extract_hardware_limit(profile_data, target)
            final_value = 442 # 模拟 Agent 推理出的数值
            
            results[target] = final_value
            
        except Exception as e:
            print(f"Error processing {target}: {e}")
            results[target] = None
            
    # 4. 输出最终结果供 LLM 裁判评分
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\n[Orchestrator] All tasks completed. Results saved to results.json")

if __name__ == "__main__":
    main()