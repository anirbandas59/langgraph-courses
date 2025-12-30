"""Pretty printing utilities for terminal output."""

from typing import Any, Dict

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_step(step_name: str, message: str, style: str = "cyan") -> None:
    """
    Print a workflow step with formatting.

    Args:
        step_name: Name of the step (e.g., "ROUTE QUESTION")
        message: Message to display
        style: Rich style for the message
    """
    console.print(f"[bold {style}]→ {step_name}:[/bold {style}] {message}")


def print_final_result(result: Dict[str, Any]) -> None:
    """
    Format and display the final result beautifully.

    Args:
        result: Dictionary containing question, generation, web_search, and documents
    """
    console.print("\n")

    # Question Panel
    console.print(
        Panel(
            f"[bold white]{result['question']}[/bold white]",
            title="❓ Question",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Answer Panel with Markdown rendering
    console.print(
        Panel(
            Markdown(result["generation"]),
            title="✨ Answer",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Metadata Table
    table = Table(title="📊 Workflow Metadata", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="yellow")

    # Web Search Status
    web_search_icon = "✅" if result["web_search"] else "❌"
    table.add_row("Web Search Used", f"{web_search_icon} {'Yes' if result['web_search'] else 'No'}")

    # Documents Count
    doc_count = len(result["documents"])
    table.add_row("Documents Retrieved", str(doc_count))

    # Sources (unique URLs)
    if result["documents"]:
        sources = set()
        for doc in result["documents"]:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                sources.add(doc.metadata["source"])

        if sources:
            # Show first 2 sources, truncate if more
            source_list = list(sources)[:2]
            source_text = "\n".join(source_list)
            if len(sources) > 2:
                source_text += f"\n... and {len(sources) - 2} more"
            table.add_row("Sources", source_text)

    console.print(table)
    console.print("\n")


def print_header() -> None:
    """Print the application header."""
    console.print("\n")
    console.print(
        Panel(
            "[bold magenta]🤖 Adaptive RAG Workflow[/bold magenta]\n"
            "[dim]Intelligent question-answering with quality validation[/dim]",
            border_style="magenta",
            padding=(1, 2),
        )
    )
    console.print("\n")


def print_workflow_start(question: str) -> None:
    """
    Print workflow start message.

    Args:
        question: The question being processed
    """
    console.print(f"[bold cyan]🔄 Processing:[/bold cyan] {question}\n")


def print_error(error_message: str) -> None:
    """
    Print an error message.

    Args:
        error_message: Error message to display
    """
    console.print(
        Panel(
            f"[bold red]{error_message}[/bold red]",
            title="❌ Error",
            border_style="red",
            padding=(1, 2),
        )
    )


def print_success(message: str) -> None:
    """
    Print a success message.

    Args:
        message: Success message to display
    """
    console.print(f"[bold green]✓[/bold green] {message}")
