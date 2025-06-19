#!/usr/bin/env python3

import logging
from .beautifier import CodeQualityManager
from .documenter import SphinxDocumenter
import asyncio
import click
from internal.resource_manager import get_resource_manager
from pathlib import Path
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.tree import Tree


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
console = Console()


class ExporterUI:
    """Interactive UI for the Exporter tool suite."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.resource_manager = get_resource_manager(self.repo_path / "cache")
        self.code_quality = CodeQualityManager(self.repo_path, self.resource_manager)
        self.documenter = SphinxDocumenter(str(self.repo_path))

    def display_menu_tree(self):
        """Display the menu as a tree structure."""
        tree = Tree(
            "🛠️  Exporter Tools", style="bold cyan", guide_style="bold bright_black"
        )

        # Full Setup
        full_setup = tree.add("📦 [bold green]Full Project Setup[/]")
        full_setup.add("   • Code Quality + Documentation")

        # Code Quality
        code_quality = tree.add("🔍 [bold yellow]Code Quality Tools[/]")
        code_quality.add("   • Run All Quality Checks")
        code_quality.add("   • Setup Pre-commit Hooks")
        code_quality.add("   • Individual Tools (Black, isort, Ruff, MyPy, Bandit)")
        code_quality.add("   • Configure Tool Settings")

        # Documentation
        docs = tree.add("📚 [bold blue]Documentation Tools[/]")
        docs.add("   • Generate Full Documentation")
        docs.add("   • Generate API Documentation")
        docs.add("   • Setup Basic Structure")
        docs.add("   • Configure Sphinx Settings")

        # Custom Setup
        custom = tree.add("⚙️  [bold magenta]Custom Setup[/]")
        custom.add("   • Mix and match components")
        custom.add("   • Choose specific tools")

        console.print("\n")
        console.print(
            Panel(
                tree, title="[bold]Welcome to Exporter[/]", border_style="bright_black"
            )
        )
        console.print("\n")

    async def main_menu(self):
        """Display the main interactive menu."""
        while True:
            # Clear screen and show tree
            console.clear()
            self.display_menu_tree()

            choice = await questionary.select(
                "Select an option:",
                choices=[
                    questionary.Choice("1. Full Project Setup", "full"),
                    questionary.Choice("2. Code Quality Tools", "quality"),
                    questionary.Choice("3. Documentation Tools", "docs"),
                    questionary.Choice("4. Custom Setup", "custom"),
                    questionary.Choice("5. Exit", "exit"),
                ],
                style=questionary.Style(
                    [
                        ("selected", "fg:white bg:blue"),
                        ("pointer", "fg:cyan bold"),
                        ("highlighted", "fg:cyan bold"),
                    ]
                ),
            ).ask_async()

            if choice == "exit":
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            elif choice == "full":
                await self.full_setup()
            elif choice == "quality":
                await self.code_quality_menu()
            elif choice == "docs":
                await self.documentation_menu()
            elif choice == "custom":
                await self.custom_setup()

            # Pause before refreshing menu
            await questionary.text("Press Enter to continue...").ask_async()

    async def full_setup(self):
        """Run full project setup with both code quality and documentation."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Code Quality Setup
                task1 = progress.add_task(
                    "[cyan]Setting up code quality tools...", total=None
                )
                await self.code_quality.setup_tool_configurations()
                await self.code_quality.run_all_checks()
                progress.update(task1, completed=True)

                # Documentation Setup
                task2 = progress.add_task(
                    "[cyan]Setting up documentation...", total=None
                )
                await self.documenter.ensure_sphinx_packages()
                await self.documenter.setup_sphinx_docs()
                progress.update(task2, completed=True)

            console.print(
                "\n[green]✓[/green] Full project setup completed successfully!"
            )

        except Exception as e:
            console.print(f"\n[red]Error during full setup: {e}[/red]")

    async def code_quality_menu(self):
        """Display the code quality tools menu."""
        while True:
            console.clear()
            tree = Tree("🔍 Code Quality Tools", style="bold yellow")
            tree.add("1. Run Comprehensive Quality Scan")
            tree.add("2. Setup Pre-commit Hooks")
            tree.add("3. Run Individual Tools")
            tree.add("4. Configure Tool Settings")
            tree.add("5. Format Files (Powered by globaltools)")
            tree.add("6. Back to Main Menu")
            console.print(Panel(tree, border_style="yellow"))

            choice = await questionary.select(
                "Select an option:",
                choices=[
                    "1. Run Comprehensive Quality Scan",
                    "2. Setup Pre-commit Hooks",
                    "3. Run Individual Tools",
                    "4. Configure Tool Settings",
                    "5. Format Files (Powered by globaltools)",
                    "6. Back to Main Menu",
                ],
                style=questionary.Style(
                    [
                        ("selected", "fg:white bg:yellow"),
                        ("pointer", "fg:yellow bold"),
                    ]
                ),
            ).ask_async()

            if "1." in choice:
                # Run comprehensive scan with new display
                success = await self.code_quality.display_comprehensive_scan()

                # Don't clear screen here - the results should remain visible
                # Instead, user will press Enter in the display_comprehensive_scan interface
                # No need for additional waiting prompt as it's already in the beautifier.py

                # Only clear when returning to menu choice
                console.clear()

            elif "2." in choice:
                await self.code_quality.setup_pre_commit()
                await questionary.text("Press Enter to continue...").ask_async()
            elif "3." in choice:
                await self.run_individual_tools()
                await questionary.text("Press Enter to continue...").ask_async()
            elif "4." in choice:
                await self.configure_tools()
                await questionary.text("Press Enter to continue...").ask_async()
            elif "5." in choice:
                await self.format_files_menu()
                await questionary.text("Press Enter to continue...").ask_async()
            elif "6." in choice:
                break

    async def documentation_menu(self):
        """Display the documentation tools menu."""
        console.clear()
        tree = Tree("📚 Documentation Tools", style="bold blue")
        tree.add("1. Generate Full Documentation")
        tree.add("2. Generate API Documentation Only")
        tree.add("3. Setup Basic Documentation Structure")
        tree.add("4. Configure Sphinx Settings")
        tree.add("5. Back to Main Menu")
        console.print(Panel(tree, border_style="blue"))

        choice = await questionary.select(
            "Select an option:",
            choices=[
                "1. Generate Full Documentation",
                "2. Generate API Documentation Only",
                "3. Setup Basic Documentation Structure",
                "4. Configure Sphinx Settings",
                "5. Back to Main Menu",
            ],
            style=questionary.Style(
                [
                    ("selected", "fg:white bg:blue"),
                    ("pointer", "fg:blue bold"),
                ]
            ),
        ).ask_async()

        if "1." in choice:
            await self.documenter.ensure_sphinx_packages()
            await self.documenter.setup_sphinx_docs()
        elif "2." in choice:
            await self.generate_api_docs()
        elif "3." in choice:
            await self.setup_basic_docs()
        elif "4." in choice:
            await self.configure_sphinx()

    async def custom_setup(self):
        """Allow users to create a custom setup configuration."""
        console.clear()
        tree = Tree("⚙️  Custom Setup", style="bold magenta")
        tree.add("[bold]Available Components:[/]")
        components = tree.add("Select multiple options:")
        components.add("• Code Formatting (Black & isort)")
        components.add("• Linting (Ruff, MyPy)")
        components.add("• Security Checks (Bandit)")
        components.add("• Pre-commit Hooks")
        components.add("• Documentation Components")
        console.print(Panel(tree, border_style="magenta"))

        options = await questionary.checkbox(
            "Choose components to set up:",
            choices=[
                questionary.Choice("Code Formatting (Black & isort)", checked=True),
                questionary.Choice("Linting (Ruff, MyPy)", checked=True),
                questionary.Choice("Security Checks (Bandit)", checked=True),
                questionary.Choice("Pre-commit Hooks", checked=True),
                questionary.Choice("Basic Documentation Structure", checked=True),
                questionary.Choice("API Documentation", checked=True),
                questionary.Choice("Full Documentation", checked=False),
            ],
            style=questionary.Style(
                [
                    ("selected", "fg:white bg:magenta"),
                    ("pointer", "fg:magenta bold"),
                    ("checkbox", "fg:magenta"),
                    ("highlighted", "fg:magenta bold"),
                ]
            ),
        ).ask_async()

        if options:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                for option in options:
                    task = progress.add_task(
                        f"[magenta]Setting up {option}...", total=None
                    )
                    if "Code Formatting" in option:
                        await self.code_quality.run_formatter("black", [])
                        await self.code_quality.run_formatter("isort", [])
                    elif "Linting" in option:
                        await self.code_quality.run_linter("ruff", [])
                        await self.code_quality.run_linter("mypy", [])
                    elif "Security Checks" in option:
                        await self.code_quality.run_linter("bandit", [])
                    elif "Pre-commit Hooks" in option:
                        await self.code_quality.setup_pre_commit()
                    elif "Basic Documentation Structure" in option:
                        await self.setup_basic_docs()
                    elif "API Documentation" in option:
                        await self.generate_api_docs()
                    elif "Full Documentation" in option:
                        await self.documenter.setup_sphinx_docs()
                    progress.update(task, completed=True)

            console.print("\n[bold green]✓[/] Custom setup completed successfully!")

    async def run_individual_tools(self):
        """Run individual code quality tools."""
        tool = await questionary.select(
            "Select a tool to run:",
            choices=[
                "Black (Code Formatter)",
                "isort (Import Formatter)",
                "Ruff (Linter)",
                "MyPy (Type Checker)",
                "Bandit (Security Checker)",
            ],
        ).ask_async()

        try:
            if "Black" in tool:
                await self.code_quality.run_formatter("black", [])
            elif "isort" in tool:
                await self.code_quality.run_formatter("isort", [])
            elif "Ruff" in tool:
                await self.code_quality.run_linter("ruff", [])
            elif "MyPy" in tool:
                await self.code_quality.run_linter("mypy", [])
            elif "Bandit" in tool:
                await self.code_quality.run_linter("bandit", [])

        except Exception as e:
            console.print(f"\n[red]Error running {tool}: {e}[/red]")

    async def configure_tools(self):
        """Configure individual tool settings."""
        tool = await questionary.select(
            "Select a tool to configure:",
            choices=["Black", "isort", "Ruff", "MyPy", "Bandit", "Pre-commit"],
        ).ask_async()

        # Show current configuration and allow modifications
        await self.code_quality.setup_tool_configurations()
        console.print(f"\n[green]✓[/green] Configuration updated for {tool}")

    async def generate_api_docs(self):
        """Generate API documentation only."""
        try:
            await self.documenter.ensure_sphinx_packages()
            # Run sphinx-apidoc only
            proc = await asyncio.create_subprocess_exec(
                "sphinx-apidoc",
                "-o",
                str(self.documenter.docs_source_path / "api"),
                "-f",
                "-e",
                "-M",
                "--implicit-namespaces",
                str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            console.print(
                "\n[green]✓[/green] API documentation generated successfully!"
            )

        except Exception as e:
            console.print(f"\n[red]Error generating API documentation: {e}[/red]")

    async def setup_basic_docs(self):
        """Set up basic documentation structure without full generation."""
        try:
            await self.documenter.ensure_sphinx_packages()
            # Create directory structure only
            await self.documenter._ensure_directory(self.documenter.docs_path)
            await self.documenter._ensure_directory(self.documenter.docs_source_path)
            await self.documenter._ensure_directory(
                self.documenter.docs_source_path / "_static"
            )
            await self.documenter._ensure_directory(
                self.documenter.docs_source_path / "_templates"
            )
            await self.documenter._ensure_directory(
                self.documenter.docs_source_path / "api"
            )

            # Create basic configuration
            await self.documenter._create_sphinx_config()
            console.print("\n[green]✓[/green] Basic documentation structure created!")

        except Exception as e:
            console.print(f"\n[red]Error setting up basic documentation: {e}[/red]")

    async def configure_sphinx(self):
        """Configure Sphinx documentation settings."""
        try:
            # Allow configuration of basic Sphinx settings
            project_name = await questionary.text(
                "Project name:", default=self.repo_path.name
            ).ask_async()

            author = await questionary.text(
                "Author name:", default="Author"
            ).ask_async()

            # Update configuration
            await self.documenter._create_sphinx_config()
            console.print("\n[green]✓[/green] Sphinx configuration updated!")

        except Exception as e:
            console.print(f"\n[red]Error configuring Sphinx: {e}[/red]")

    async def format_files_menu(self):
        """Display menu for file formatting options."""
        console.clear()
        tree = Tree("🔧 Format Files", style="bold cyan")
        tree.add("1. Format Current Directory")
        tree.add("2. Format Specific File")
        tree.add("3. Format Specific Directory")
        tree.add("4. Format with Specific Tool")
        tree.add("5. Back to Code Quality Menu")
        console.print(Panel(tree, border_style="cyan"))

        choice = await questionary.select(
            "Select an option:",
            choices=[
                "1. Format Current Directory",
                "2. Format Specific File",
                "3. Format Specific Directory",
                "4. Format with Specific Tool",
                "5. Back to Code Quality Menu",
            ],
            style=questionary.Style(
                [
                    ("selected", "fg:white bg:cyan"),
                    ("pointer", "fg:cyan bold"),
                ]
            ),
        ).ask_async()

        if "1." in choice:
            # Format current directory
            console.print("[cyan]Running formatter on current directory...[/cyan]")
            import subprocess

            try:
                # Call the external formatter from the exporter script
                process = await asyncio.create_subprocess_exec(
                    "./exporter",
                    "format",
                    ".",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Stream output in real-time
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    console.print(line.decode().strip())

                await process.wait()

                if process.returncode == 0:
                    console.print("[green]Formatting completed successfully[/green]")
                else:
                    stderr = await process.stderr.read()
                    console.print(f"[red]Formatting failed: {stderr.decode()}[/red]")

            except Exception as e:
                console.print(f"[red]Error running formatter: {e!s}[/red]")

        elif "2." in choice:
            # Format specific file
            file_path = await questionary.text(
                "Enter file path to format:", default=str(self.repo_path)
            ).ask_async()

            if file_path:
                console.print(f"[cyan]Formatting file: {file_path}[/cyan]")
                import subprocess

                try:
                    process = await asyncio.create_subprocess_exec(
                        "./exporter",
                        "format",
                        file_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        console.print(line.decode().strip())

                    await process.wait()
                except Exception as e:
                    console.print(f"[red]Error formatting file: {e!s}[/red]")

        elif "3." in choice:
            # Format specific directory
            dir_path = await questionary.text(
                "Enter directory path to format:", default=str(self.repo_path)
            ).ask_async()

            recursive = await questionary.confirm(
                "Format recursively (include subdirectories)?", default=True
            ).ask_async()

            if dir_path:
                console.print(
                    f"[cyan]Formatting directory: {dir_path} {'(recursive)' if recursive else ''}[/cyan]"
                )

                try:
                    cmd = ["./exporter", "format", dir_path]
                    if not recursive:
                        cmd.append("--no-recursive")

                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        console.print(line.decode().strip())

                    await process.wait()
                except Exception as e:
                    console.print(f"[red]Error formatting directory: {e!s}[/red]")

        elif "4." in choice:
            # Format with a specific tool
            tools = [
                "black",
                "isort",
                "prettier",
                "ruff",
                "clang-format",
                "gofmt",
                "shfmt",
                "rustfmt",
            ]

            tool = await questionary.select(
                "Select formatting tool:", choices=tools
            ).ask_async()

            target = await questionary.text(
                "Enter path to format:", default=str(self.repo_path)
            ).ask_async()

            if tool and target:
                console.print(f"[cyan]Running {tool} on {target}[/cyan]")

                try:
                    # Check if the tool is installed
                    import shutil

                    if not shutil.which(tool):
                        # Try to install with globaltools
                        console.print(
                            f"[yellow]{tool} not found. Attempting to install...[/yellow]"
                        )
                        install_process = await asyncio.create_subprocess_exec(
                            "globaltools",
                            "install",
                            tool,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        await install_process.wait()

                    # Run the tool directly
                    # Mapping of tool-specific arguments
                    tool_args = {
                        "black": ["--quiet"],
                        "isort": ["--profile=black"],
                        "prettier": ["--write"],
                        "ruff": ["check", "--fix"],
                        "clang-format": ["-i"],
                        "gofmt": ["-w"],
                        "shfmt": ["-w"],
                        "rustfmt": [],
                    }

                    # Run the tool with appropriate args
                    args = [tool]
                    args.extend(tool_args.get(tool, []))
                    args.append(target)

                    process = await asyncio.create_subprocess_exec(
                        *args,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    stdout, stderr = await process.communicate()

                    if process.returncode == 0:
                        console.print(f"[green]{tool} completed successfully[/green]")
                        if stdout:
                            console.print(stdout.decode())
                    else:
                        console.print(f"[red]{tool} failed:[/red] {stderr.decode()}")

                except Exception as e:
                    console.print(f"[red]Error running {tool}: {e!s}[/red]")

        elif "5." in choice:
            # Return to previous menu
            return


@click.command()
@click.option("--repo-path", default=".", help="Path to the repository")
def main(repo_path: str):
    """Exporter UI - Interactive tool for code quality and documentation."""
    try:
        console.clear()
        ui = ExporterUI(repo_path)
        asyncio.run(ui.main_menu())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 👋[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
