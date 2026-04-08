from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.spinner import Spinner
from core.state import WorkflowState

class UIRenderer:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="upper", ratio=1),
            Layout(name="lower", ratio=1)
        )
        self.layout["upper"].split_row(
            Layout(name="strategy", ratio=1),
            Layout(name="executor", ratio=1)
        )
        self.current_executor_status = "Initializing..."
        
    def update_executor_status(self, status: str):
        self.current_executor_status = status

    def render(self, state: WorkflowState) -> Layout:
        # 1. Current Probe Strategy
        strategy_text = "Active Targets:\n"
        if not state.targets:
            strategy_text += "No targets loaded yet."
        for t in state.targets:
            strategy_text += f"- [cyan]{t.get('name')}[/cyan] (Strategy: [magenta]{t.get('type')}[/magenta])\n"
        self.layout["strategy"].update(
            Panel(strategy_text, title="[bold blue]Probe Strategy[/bold blue]", border_style="blue")
        )
        
        # 2. Executor State with Loading Animation
        spinner = Spinner("dots", text=self.current_executor_status)
        self.layout["executor"].update(
            Panel(spinner, title="[bold yellow]Executor State[/bold yellow]", border_style="yellow")
        )
        
        # 3. System Reasoning Stream
        # Display the last 15 lines of reasoning
        reasoning_lines = state.reasoning_history[-15:] if state.reasoning_history else ["Waiting for reasoning logs..."]
        reasoning_text = "\n".join(reasoning_lines)
        self.layout["lower"].update(
            Panel(reasoning_text, title="[bold green]Agent Reasoning Stream[/bold green]", border_style="green")
        )
        
        return self.layout
