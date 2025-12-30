"""Main entry point for the Adaptive RAG Workflow."""

import argparse

from dotenv import load_dotenv
from rich.prompt import Prompt

from graph.graph import app
from utils import (
    print_error,
    print_final_result,
    print_header,
    print_workflow_start,
    setup_logging,
)

load_dotenv()


def run_query(question: str, verbose: bool = False) -> None:
    """
    Run a single query through the RAG workflow.

    Args:
        question: The question to ask
        verbose: If True, show detailed workflow steps
    """
    try:
        print_workflow_start(question)
        result = app.invoke(input={"question": question})
        print_final_result(result)
    except Exception as e:
        print_error(f"Failed to process question: {str(e)}")


def interactive_mode(verbose: bool = False) -> None:
    """
    Run the application in interactive mode.

    Args:
        verbose: If True, show detailed workflow steps
    """
    print_header()
    print("[dim]Type 'quit', 'exit', or 'q' to exit interactive mode.[/dim]\n")

    while True:
        try:
            question = Prompt.ask("\n[cyan]❓ Ask a question[/cyan]")

            if question.lower() in ["quit", "exit", "q"]:
                print("\n[yellow]👋 Goodbye![/yellow]\n")
                break

            if not question.strip():
                print_error("Question cannot be empty. Please try again.")
                continue

            run_query(question, verbose)

        except KeyboardInterrupt:
            print("\n\n[yellow]👋 Goodbye![/yellow]\n")
            break
        except Exception as e:
            print_error(f"An error occurred: {str(e)}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Adaptive RAG Workflow - Intelligent question-answering system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a single question
  python main.py -q "What is agent memory?"

  # Interactive mode
  python main.py -i

  # Verbose output with detailed workflow steps
  python main.py -q "What is prompt engineering?" -v

  # Interactive mode with verbose output
  python main.py -i -v
        """,
    )

    parser.add_argument(
        "-q",
        "--question",
        type=str,
        help="Question to ask (single query mode)",
    )

    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode (ask multiple questions)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (show detailed workflow steps)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: WARNING)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, suppress_warnings=True)

    # Determine mode
    if args.interactive:
        # Interactive mode
        interactive_mode(verbose=args.verbose)
    elif args.question:
        # Single question mode
        print_header()
        run_query(args.question, verbose=args.verbose)
    else:
        # No arguments provided - show help and run default question
        print_header()
        print("[dim]No arguments provided. Running with default question...[/dim]")
        print('[dim]Use --help to see all available options.[/dim]\n')
        run_query("What is agent memory?", verbose=args.verbose)


if __name__ == "__main__":
    main()
