from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class ExecutionResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    success: bool = False
    metrics: Dict[str, Any] = Field(default_factory=dict)

class WorkflowState(BaseModel):
    targets: List[Dict[str, Any]] = Field(default_factory=list)
    executable_path: Optional[str] = None
    generated_code_paths: List[str] = Field(default_factory=list)
    execution_results: List[ExecutionResult] = Field(default_factory=list)
    extracted_metrics: Dict[str, Any] = Field(default_factory=dict)
    reasoning_history: List[str] = Field(default_factory=list)
    
    def add_reasoning(self, step: str):
        self.reasoning_history.append(step)
