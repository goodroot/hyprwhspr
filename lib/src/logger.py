"""
Centralized logging system for hyprwhspr using rich for beautiful CLI output
"""

import sys
from typing import Optional

# Try to import rich, with fallback to basic print functionality
try:
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.prompt import Confirm
    from rich import box
    RICH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RICH_AVAILABLE = False
    # Create minimal fallback classes
    class Console:
        def __init__(self, stderr=False):
            self.stderr = stderr
        def print(self, *args, **kwargs):
            if self.stderr:
                print(*args, file=sys.stderr, **kwargs)
            else:
                print(*args, **kwargs)
        def rule(self, title="", style=""):
            print("=" * 60)
            if title:
                print(title)
    
    class Text:
        def __init__(self):
            self.parts = []
        def append(self, text, style=None):
            self.parts.append(text)
        def __str__(self):
            return "".join(self.parts)
    
    class Panel:
        def __init__(self, content, box=None, style=None, padding=None):
            self.content = content
    class Progress:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def add_task(self, *args, **kwargs):
            pass
    class SpinnerColumn:
        pass
    class TextColumn:
        pass
    class Table:
        def __init__(self, title=None, box=None):
            self.title = title
            self.headers = []
            self.rows = []
        def add_column(self, header, style=None):
            self.headers.append(header)
        def add_row(self, *cells):
            self.rows.append(cells)
    class Confirm:
        @staticmethod
        def ask(question, default=False, console=None):
            prompt = f"{question} [Y/n]: " if default else f"{question} [y/N]: "
            try:
                response = input(prompt).strip().lower()
                if not response:
                    return default
                return response in ('y', 'yes')
            except (EOFError, KeyboardInterrupt):
                return default
    
    # Create a simple box namespace
    class BoxNamespace:
        ROUNDED = ""
        SIMPLE_HEAVY = ""
    box = BoxNamespace()


class WhisperLogger:
    """Centralized logger with rich formatting for consistent CLI output"""
    
    def __init__(self):
        self.console = Console()
        self.error_console = Console(stderr=True)
        
    def info(self, message: str, prefix: str = "INFO"):
        """Log info message with blue styling"""
        if RICH_AVAILABLE:
            text = Text()
            text.append(f"[{prefix}] ", style="bold blue")
            text.append(message)
            self.console.print(text)
        else:
            print(f"[{prefix}] {message}")
    
    def success(self, message: str, prefix: str = "SUCCESS"):
        """Log success message with green styling"""
        if RICH_AVAILABLE:
            text = Text()
            text.append("SUCCESS: ", style="bold green")
            text.append(f"[{prefix}] ", style="bold green")
            text.append(message)
            self.console.print(text)
        else:
            print(f"SUCCESS: [{prefix}] {message}")
    
    def warning(self, message: str, prefix: str = "WARNING"):
        """Log warning message with yellow styling"""
        if RICH_AVAILABLE:
            text = Text()
            text.append("WARNING: ", style="bold yellow")
            text.append(f"[{prefix}] ", style="bold yellow")
            text.append(message)
            self.console.print(text)
        else:
            print(f"WARNING: [{prefix}] {message}", file=sys.stderr)
    
    def error(self, message: str, prefix: str = "ERROR"):
        """Log error message with red styling"""
        if RICH_AVAILABLE:
            text = Text()
            text.append("ERROR: ", style="bold red")
            text.append(f"[{prefix}] ", style="bold red")
            text.append(message)
            self.error_console.print(text)
        else:
            print(f"ERROR: [{prefix}] {message}", file=sys.stderr)
    
    def step(self, message: str, prefix: str = "STEP"):
        """Log step message with arrow styling"""
        if RICH_AVAILABLE:
            text = Text()
            text.append("→ ", style="bold cyan")
            text.append(f"[{prefix}] ", style="bold cyan")
            text.append(message)
            self.console.print(text)
        else:
            print(f"→ [{prefix}] {message}")
    
    def debug(self, message: str, prefix: str = "DEBUG"):
        """Log debug message with dim styling"""
        if RICH_AVAILABLE:
            text = Text()
            text.append(f"[{prefix}] ", style="dim")
            text.append(message, style="dim")
            self.console.print(text)
        else:
            print(f"[{prefix}] {message}")
    
    def header(self, title: str, subtitle: Optional[str] = None):
        """Print a formatted header"""
        if RICH_AVAILABLE:
            if subtitle:
                panel_content = f"[bold]{title}[/bold]\n{subtitle}"
            else:
                panel_content = f"[bold]{title}[/bold]"
                
            panel = Panel(
                panel_content,
                box=box.ROUNDED,
                style="blue",
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
            print("\n" + "=" * 60)
            print(title)
            if subtitle:
                print(subtitle)
            print("=" * 60)
    
    def section(self, title: str):
        """Print a section divider"""
        if RICH_AVAILABLE:
            self.console.print(f"\n[bold blue]═══ {title} ═══[/bold blue]")
        else:
            print(f"\n═══ {title} ═══")
    
    def table(self, title: str, headers: list, rows: list):
        """Print a formatted table"""
        if RICH_AVAILABLE:
            table = Table(title=title, box=box.SIMPLE_HEAVY)
            
            for header in headers:
                table.add_column(header, style="cyan")
            
            for row in rows:
                table.add_row(*[str(cell) for cell in row])
            
            self.console.print(table)
        else:
            if title:
                print(f"\n{title}")
            print(" | ".join(headers))
            print("-" * (sum(len(str(h)) for h in headers) + 3 * (len(headers) - 1)))
            for row in rows:
                print(" | ".join(str(cell) for cell in row))
    
    def progress_context(self, description: str = "Processing..."):
        """Context manager for progress spinner"""
        if RICH_AVAILABLE:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            )
        else:
            # Return a no-op context manager
            class NoOpProgress:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def add_task(self, *args, **kwargs):
                    pass
            return NoOpProgress()
    
    def ask_confirmation(self, question: str, default: bool = False) -> bool:
        """Ask user for confirmation"""
        return Confirm.ask(question, default=default, console=self.console)
    
    def rule(self, title: str = ""):
        """Print a horizontal rule"""
        if RICH_AVAILABLE:
            self.console.rule(title, style="blue")
        else:
            print("=" * 60)
            if title:
                print(title)


# Create global logger instance
logger = WhisperLogger()


# Convenience functions for easy importing
def log_info(message: str, prefix: str = "INFO"):
    logger.info(message, prefix)

def log_success(message: str, prefix: str = "SUCCESS"):
    logger.success(message, prefix)

def log_warning(message: str, prefix: str = "WARNING"):
    logger.warning(message, prefix)

def log_error(message: str, prefix: str = "ERROR"):
    logger.error(message, prefix)

def log_step(message: str, prefix: str = "STEP"):
    logger.step(message, prefix)

def log_debug(message: str, prefix: str = "DEBUG"):
    logger.debug(message, prefix)
