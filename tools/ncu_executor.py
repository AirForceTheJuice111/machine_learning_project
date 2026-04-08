import subprocess
import json

class NcuExecutor:
    def run_with_ncu(self, exe_path: str, metrics: list) -> dict:
        """运行可执行文件，并使用 ncu 抓取特定指标"""
        metrics_str = ",".join(metrics)
        # --csv 参数方便后续代码解析，--page details 获取详细指标
        ncu_cmd = [
            "ncu", "--metrics", metrics_str, "--csv", "--page", "details", exe_path
        ]
        
        try:
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, check=True)
            # 这里可以添加一个简单的 CSV 解析逻辑，将结果转为字典
            # 为了演示，直接返回原生输出
            return {"success": True, "raw_output": result.stdout}
        except subprocess.CalledProcessError as e:
            return {"success": False, "raw_output": e.stderr}