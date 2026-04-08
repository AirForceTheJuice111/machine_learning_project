from abc import ABC, abstractmethod
from typing import Dict, Type

class ProbeStrategy(ABC):
    @abstractmethod
    def generate_cuda_code(self, target: dict) -> str:
        """Generate CUDA code string for the given target."""
        pass
        
    @abstractmethod
    def parse_ncu_output(self, stdout: str) -> dict:
        """Parse ncu output to extract metrics."""
        pass

class ProbeRegistry:
    _registry: Dict[str, Type[ProbeStrategy]] = {}
    
    @classmethod
    def register(cls, target_type: str):
        def decorator(strategy_cls: Type[ProbeStrategy]):
            cls._registry[target_type] = strategy_cls
            return strategy_cls
        return decorator
        
    @classmethod
    def get_strategy(cls, target_type: str) -> ProbeStrategy:
        strategy_cls = cls._registry.get(target_type)
        if not strategy_cls:
            raise ValueError(f"No probe strategy registered for target_type: {target_type}")
        return strategy_cls()
