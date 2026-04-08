import subprocess
import os

class CudaCompiler:
    def __init__(self, workspace_dir="workspace"):
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)

    def compile(self, source_code: str, filename: str = "probe.cu") -> dict:
        """接收 Agent 生成的 CUDA 代码并编译，返回是否成功及报错信息"""
        filepath = os.path.join(self.workspace_dir, filename)
        exe_path = os.path.join(self.workspace_dir, filename.replace(".cu", ""))
        
        with open(filepath, "w") as f:
            f.write(source_code)
            
        # 使用 nvcc 编译
        compile_cmd = ["nvcc", "-O3", filepath, "-o", exe_path]
        try:
            result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
            return {"success": True, "exe_path": exe_path, "message": "Compilation successful."}
        except subprocess.CalledProcessError as e:
            # 编译失败时，将错误信息返回给 Agent 进行自修复 (Self-correction)
            return {"success": False, "exe_path": None, "message": e.stderr}