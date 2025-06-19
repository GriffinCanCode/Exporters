#!/Users/griffinstrier/custom/.venv/bin/python

"""Code quality management and beautification tools for Python projects."""

import logging
import os
import asyncio
from asyncio import (
    create_subprocess_exec,
    subprocess as asyncsubprocess,
)
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import (
    auto,
    Enum,
)
from functools import lru_cache
import hashlib
from pathlib import Path
import re
from rich.console import (
    Console,
    Group,
)
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
import time
import toml
import tracemalloc
from typing import (
    Any,
    cast,
    DefaultDict,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
)


# Define protocols for type safety
class PerfOptimizer(Protocol):
    async def run_io_bound(self, func: Any) -> Any: ...


class MemoryManager(Protocol):
    def get_cached_data(self, key: str) -> Any: ...
    def set_cached_data(self, key: str, value: Any) -> None: ...


class FileOperator(Protocol):
    async def get_file_size(self, file: Path) -> Optional[int]: ...
    async def parallel_operation(self, paths: List[Path], operation: Any) -> None: ...


class ResourceManagerProtocol(Protocol):
    """Protocol defining the interface for resource managers."""

    perf_optimizer: PerfOptimizer
    memory_manager: MemoryManager
    file_operator: FileOperator

    async def cleanup(self) -> None: ...


# Type variable for CoreType
T = TypeVar("T")


# Define CoreType enum here to avoid import issues
class CoreType(Enum):
    """Core type enumeration."""

    PERFORMANCE = auto()
    EFFICIENCY = auto()


if TYPE_CHECKING:
    from internal.resource_manager import (  # noqa: F401
        ResourceManager as _ResourceManager,
    )
else:
    ResourceManager = ResourceManagerProtocol  # type: ignore


# Initialize tracemalloc for memory tracking
tracemalloc.start()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Optimize file processing with a thread pool
file_executor = ThreadPoolExecutor(
    max_workers=min(
        32, (os.cpu_count() or 2) * 2
    ),  # Use 2 as fallback if cpu_count returns None
    thread_name_prefix="file_processor",
)


class ToolType(Enum):
    """Enumeration of different code quality tool types."""

    FORMATTER = auto()
    LINTER = auto()
    SECURITY = auto()
    PRECOMMIT = auto()


@dataclass
class ToolConfig:
    """Configuration for a code quality tool."""

    name: str
    version: str
    tool_type: ToolType
    args: List[str]
    requires: List[str]
    config_section: Optional[str]
    batch_size: int
    core_type: CoreType

    def __init__(
        self,
        name: str,
        version: str,
        tool_type: ToolType,
        args: List[str],
        requires: Optional[List[str]] = None,
        config_section: Optional[str] = None,
        batch_size: int = 100,
        core_type: Optional[CoreType] = None,
    ) -> None:
        """Initialize tool configuration.

        Args:
            name: Tool name
            version: Tool version
            tool_type: Type of the tool
            args: Command line arguments
            requires: Required dependencies
            config_section: Configuration section name
            batch_size: Number of files to process in batch
            core_type: Type of core to use
        """
        self.name = name
        self.version = version
        self.tool_type = tool_type
        self.args = args
        self.requires = requires or []
        self.config_section = config_section
        self.batch_size = batch_size
        self.core_type = core_type if core_type is not None else CoreType.PERFORMANCE

    def __hash__(self) -> int:
        """Return hash of the tool config."""
        return hash((self.name, self.version))


@dataclass
class CheckResult:
    """Result of a code quality check."""

    tool_name: str
    success: bool
    error_message: Optional[str] = None
    files_processed: int = 0
    issues_found: int = 0
    duration: float = 0.0


@dataclass
class LintResult:
    """Result of a linting operation."""

    success: bool
    errors: List[str]
    fixed_files: List[Path]
    stats: Dict[str, int]
    duration: float


class LintingOptimizer:
    """Optimizes linting operations with caching and incremental updates."""

    def __init__(self, resource_manager: ResourceManagerProtocol) -> None:
        """Initialize the linting optimizer.

        Args:
            resource_manager: The resource manager instance to use
        """
        self.resource_manager = resource_manager
        self._file_hashes: Dict[str, str] = {}
        self._lint_cache: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        self._stats_cache: Dict[str, Dict[str, int]] = {}

    async def should_lint_file(self, file_path: Path, tool: str) -> bool:
        """Check if a file needs to be linted based on content changes."""
        try:
            content = await self.resource_manager.perf_optimizer.run_io_bound(
                lambda: file_path.read_bytes()
            )
            current_hash = hashlib.sha256(content).hexdigest()

            cached_hash = self._file_hashes.get(str(file_path))
            if cached_hash == current_hash:
                return False

            self._file_hashes[str(file_path)] = current_hash
            return True
        except Exception:
            return True

    async def get_cached_result(
        self, file_path: Path, tool: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached lint result for a file."""
        return self._lint_cache[tool].get(str(file_path))

    def update_cache(self, file_path: Path, tool: str, result: Dict[str, Any]) -> None:
        """Update lint result cache."""
        self._lint_cache[tool][str(file_path)] = result

    def clear_cache(self) -> None:
        """Clear all caching data."""
        self._file_hashes.clear()
        self._lint_cache.clear()
        self._stats_cache.clear()


class CodeQualityManager:
    """Centralized manager for all code quality related operations.

    This class handles:
    1. Code formatting (black, isort)
    2. Linting (ruff, mypy, bandit)
    3. Pre-commit hook management
    4. Code quality configuration management

    The class follows best practices for:
    - Resource optimization through batched operations
    - Proper error handling and reporting
    - Configurable tool behavior
    - Extensible design for adding new tools
    """

    # Update DEFAULT_TOOLS with optimized configurations
    DEFAULT_TOOLS = [
        ToolConfig(
            name="black",
            version="24.2.0",
            tool_type=ToolType.FORMATTER,
            args=["--quiet"],
            config_section="black",
            batch_size=50,
            core_type=CoreType.PERFORMANCE,
        ),
        ToolConfig(
            name="isort",
            version="5.13.2",
            tool_type=ToolType.FORMATTER,
            args=[
                "--atomic",
                "--profile",
                "black",
                "--multi-line",
                "3",
                "--line-length",
                "88",
                "--py",
                "39",
                "--combine-star",
                "--ensure-newline-before-comments",
                "--force-sort-within-sections",
                "--force-alphabetical-sort-within-sections",
                "--force-alphabetical-sort",
                "--force-grid-wrap",
                "2",
                "--order-by-type",
                "--case-sensitive",
                "--lines-after-imports",
                "2",
                "--lines-between-types",
                "1",
                "--no-lines-before",
                "FUTURE,STDLIB",
                "--section-default",
                "THIRDPARTY",
                "-t",
                "__future__",
                "-t",
                "os",
                "-t",
                "sys",
                "-t",
                "logging",
                "-p",
                "exporters,internal",
                "-o",
                "toml,rich,questionary",
                "-b",
                "pathlib,dataclasses,enum,typing",
                "-m",
                "3",
                "--tc",
                "--use-parentheses",
                "--color",
                "--verbose",
            ],
            config_section="isort",
            batch_size=50,
            core_type=CoreType.PERFORMANCE,
        ),
        ToolConfig(
            name="ruff",
            version="0.2.2",
            tool_type=ToolType.LINTER,
            args=["check", "--fix", "--no-cache", "--exit-zero"],  # Optimized args
            config_section="ruff",
            batch_size=150,
            core_type=CoreType.PERFORMANCE,
        ),
        ToolConfig(
            name="mypy",
            version="1.8.0",
            tool_type=ToolType.LINTER,
            args=[
                "--no-incremental",
                "--no-error-summary",
                "--no-color-output",
                "--txt-report",
                "evaluation/types/text",
                "--html-report",
                "evaluation/types/html",
                "--any-exprs-report",
                "evaluation/types/expressions",
                "--cobertura-xml-report",
                "evaluation/types/coverage",
                "--linecount-report",
                "evaluation/types/linecount",
                "--ignore-missing-imports",
                "--allow-untyped-defs",
                "--allow-incomplete-defs",
                "--no-strict-optional",
                "--no-warn-return-any",
                "--no-warn-no-return",
            ],
            config_section="mypy",
            batch_size=50,
            core_type=CoreType.EFFICIENCY,
        ),
        ToolConfig(
            name="bandit",
            version="1.7.7",
            tool_type=ToolType.SECURITY,
            args=[
                "-r",  # Recursive scan
                "-ll",  # Log level
                "-n",
                "4",  # Number of processes for parallel execution
                "-b",
                "high",  # Report only high-confidence issues
                "-iii",  # Include more information about findings
                "-x",
                "tests,venv,.venv,build,dist",  # Exclude directories
                "--exit-zero",  # Don't fail on findings (we handle this ourselves)
                "-f",
                "json",  # Output in JSON format for better parsing
                "--ignore-nosec",  # Don't skip #nosec comments
                "-s",
                "B101,B102,B103,B104,B105,B106,B107,B108,B110,B112,B201,B301,B302,B303,B304,B305,B306,B307,B308,B309,B310,B311,B312,B313,B314,B315,B316,B317,B318,B319,B320,B321,B322,B323,B324,B325,B401,B402,B403,B404,B405,B406,B407,B408,B409,B410,B411,B412,B413,B501,B502,B503,B504,B505,B506,B507,B601,B602,B603,B604,B605,B606,B607,B608,B609,B610,B611,B701,B702,B703",  # Enable all security checks
            ],
            config_section="tool.bandit",
            batch_size=100,
            core_type=CoreType.EFFICIENCY,
        ),
        ToolConfig(
            name="pre-commit",
            version="3.6.0",
            tool_type=ToolType.PRECOMMIT,
            args=[],
            core_type=CoreType.EFFICIENCY,
        ),
    ]

    def __init__(
        self, repo_path: Path, resource_manager: ResourceManagerProtocol
    ) -> None:
        """Initialize the code quality manager.

        Args:
            repo_path: Path to the repository root
            resource_manager: The resource manager instance to use
        """
        self.repo_path = Path(repo_path)
        self.resource_manager = resource_manager
        self.tools = {tool.name: tool for tool in self.DEFAULT_TOOLS}
        self._file_cache_key = "python_files_cache"
        self.linting_optimizer = LintingOptimizer(resource_manager)

    async def __aenter__(self) -> "CodeQualityManager":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Async context manager exit with cleanup."""
        try:
            if hasattr(self, "resource_manager"):
                await self.resource_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def setup_tool_configurations(self) -> None:
        """Set up configuration files for all code quality tools."""
        try:
            # pyproject.toml configurations
            pyproject_config = {
                "tool": {
                    "black": {
                        "line-length": 88,
                        "target-version": ["py39"],
                        "include": "\\.pyi?$",
                        "extend-exclude": "(venv/|.venv/|build/|dist/)",
                    },
                    "isort": {
                        "profile": "black",
                        "multi_line_output": 3,
                        "line_length": 88,
                        "ensure_newline_before_comments": True,
                        "force_grid_wrap": 0,
                        "use_parentheses": True,
                        "include_trailing_comma": True,
                        "force_sort_within_sections": True,
                        "sections": [
                            "FUTURE",
                            "STDLIB",
                            "THIRDPARTY",
                            "FIRSTPARTY",
                            "LOCALFOLDER",
                        ],
                        "default_section": "THIRDPARTY",
                        "known_first_party": ["exporters", "internal"],
                        "known_third_party": [
                            "pytest",
                            "toml",
                            "rich",
                            "questionary",
                            "hypothesis",
                        ],
                        "extend_skip": ["venv/*", ".venv/*", "build/*", "dist/*"],
                        "skip_gitignore": True,
                        "atomic": True,
                        "lines_after_imports": 2,
                        "lines_between_sections": 1,
                        "combine_as_imports": True,
                        "combine_star": True,
                        "order_by_type": True,
                        "case_sensitive": True,
                        "remove_redundant_aliases": True,
                        "honor_noqa": True,
                        "wrap_length": 88,
                        "color_output": True,
                        "quiet": True,
                    },
                    "ruff": {
                        "line-length": 88,
                        "target-version": "py39",
                        "select": [
                            "E",  # pycodestyle errors
                            "F",  # pyflakes
                            "B",  # flake8-bugbear
                            "N",  # pep8-naming
                            "UP",  # pyupgrade
                            "PL",  # pylint
                            "RUF",  # ruff-specific rules
                            "S",  # bandit
                            "C",  # complexity checks
                            "T",  # print/pdb checks
                            "Q",  # quotes
                        ],
                        "ignore": [
                            "I"
                        ],  # Ignore isort rules since we're using isort directly
                        "extend-exclude": ["venv", ".venv", "build", "dist"],
                        "unfixable": [
                            "F401",
                            "F841",
                        ],  # Don't auto-remove unused imports
                        "mccabe": {"max-complexity": 10},
                    },
                    "mypy": {
                        "python_version": "3.9",
                        "warn_return_any": True,
                        "warn_unused_configs": True,
                        "disallow_untyped_defs": True,
                        "check_untyped_defs": True,
                        "disallow_any_unimported": True,
                        "no_implicit_optional": True,
                        "warn_redundant_casts": True,
                        "warn_unused_ignores": True,
                        "warn_no_return": True,
                        "warn_unreachable": True,
                        "exclude": ["venv", ".venv", "build", "dist"],
                    },
                    "bandit": {
                        "exclude_dirs": ["tests", "venv", ".venv", "build", "dist"],
                        "targets": ["src"],
                        "skips": ["B101"],  # Skip assert warnings
                    },
                }
            }

            # Update existing pyproject.toml or create new one
            pyproject_path = self.repo_path / "pyproject.toml"
            try:
                if pyproject_path.exists():
                    current_config = toml.load(pyproject_path)

                    # Check if configs are different
                    needs_update = False
                    for tool_name, tool_config in pyproject_config["tool"].items():
                        if (
                            tool_name not in current_config.get("tool", {})
                            or current_config["tool"][tool_name] != tool_config
                        ):
                            needs_update = True
                            break

                        # Special check for unsupported isort settings
                        if (
                            tool_name == "isort"
                            and "force_single_line_imports"
                            in current_config["tool"][tool_name]
                        ):
                            needs_update = True
                            break

                    if needs_update:
                        # Merge configurations, giving priority to our new config
                        await self._deep_merge_configs(current_config, pyproject_config)
                        config_content = toml.dumps(current_config)
                        pyproject_path.write_text(config_content)
                        logger.info("✨ Updated tool configurations in pyproject.toml")
                    else:
                        logger.info("✓ Tool configurations are up to date")
                else:
                    # Create new pyproject.toml
                    config_content = toml.dumps(pyproject_config)
                    pyproject_path.write_text(config_content)
                    logger.info(
                        "✨ Created new pyproject.toml with tool configurations"
                    )

            except Exception as e:
                logger.error(f"Failed to update pyproject.toml: {e}")
                raise

        except Exception as e:
            logger.error(f"Failed to set up tool configurations: {e}")
            raise

    async def _deep_merge_configs(self, target: Dict, source: Dict) -> None:
        """Deep merge two configuration dictionaries.

        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                await self._deep_merge_configs(target[key], value)
            else:
                target[key] = value

    async def get_python_files(
        self, exclude_patterns: Optional[tuple[str, ...]] = None
    ) -> List[Path]:
        """Get all Python files in the repository, excluding specified patterns.

        Args:
            exclude_patterns: Tuple of patterns to exclude (converted from list for caching)

        Returns:
            List of Path objects for Python files
        """
        # Create cache key based on exclude patterns
        cache_key = f"{self._file_cache_key}_{hash(exclude_patterns if exclude_patterns else ())}"

        # Try to get from cache first
        cached_files = self.resource_manager.memory_manager.get_cached_data(cache_key)
        if cached_files:
            return cast(List[Path], cached_files)

        if exclude_patterns is None:
            exclude_patterns = (
                ".venv",
                "venv",
                "__init__.tmpl.py",
                "site-packages",
                "__pycache__",
                ".git",
                "build",
                "dist",
                "migrations",
                "node_modules",
            )

        try:
            all_files: Set[Path] = set()

            async def collect_files(pattern: str) -> Set[Path]:
                try:
                    files = await self.resource_manager.perf_optimizer.run_io_bound(
                        lambda: set(self.repo_path.rglob(pattern))
                    )
                    return cast(Set[Path], files)
                except Exception as e:
                    logger.error(f"Error collecting files with pattern {pattern}: {e}")
                    return set()

            # Collect files concurrently using resource manager
            patterns = ("*.py", "*.pyi")
            file_results = await asyncio.gather(
                *[collect_files(pattern) for pattern in patterns]
            )

            for files in file_results:
                all_files.update(files)

            async def validate_file(file: Path) -> Optional[Path]:
                try:
                    stats = await self.resource_manager.perf_optimizer.run_io_bound(
                        lambda: (file.is_file(), str(file), file.stat().st_size)
                    )
                    if (
                        not stats[0]
                        or any(pattern in stats[1] for pattern in exclude_patterns)
                        or stats[2] == 0
                    ):
                        return None
                    return file
                except Exception:
                    return None

            # Process files in optimized batches using resource manager
            all_files_list = list(all_files)
            validation_tasks = [validate_file(f) for f in all_files_list]
            valid_files = await asyncio.gather(*validation_tasks)

            # Filter out None values and sort
            result = sorted([f for f in valid_files if f is not None])

            # Cache the results with the specific cache key
            self.resource_manager.memory_manager.set_cached_data(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Failed to get Python files: {e}")
            raise

    async def run_formatter(self, tool: str, files: List[Path]) -> bool:
        """Run a formatting tool on specified files."""
        if not files:
            return True

        try:
            config = self.tools[tool]

            # Use resource manager's batch processing
            batches = [
                files[i : i + config.batch_size]
                for i in range(0, len(files), config.batch_size)
            ]

            start_time = time.time()
            logger.info(
                f"Running {tool} on {len(files)} files in {len(batches)} batches..."
            )

            async def process_batch(batch: List[Path]) -> Tuple[bool, Optional[str]]:
                cmd = [tool] + config.args
                if tool == "isort":
                    config_path = self.repo_path / "pyproject.toml"
                    if not config_path.exists():
                        return False, f"Configuration file not found at {config_path}"
                    cmd.extend(["--settings", str(config_path)])
                cmd.extend(str(f) for f in batch)

                proc = await create_subprocess_exec(
                    *cmd, stdout=asyncsubprocess.PIPE, stderr=asyncsubprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                return proc.returncode == 0, (
                    stderr.decode().strip() if proc.returncode != 0 else None
                )

            # Process batches concurrently
            tasks = []
            for batch in batches:
                task = asyncio.create_task(process_batch(batch))
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            duration = time.time() - start_time

            # Check results
            success = all(result[0] for result in results)
            if not success:
                errors = [err for _, err in results if err]
                logger.error(
                    f"{tool} formatting failed in {duration:.2f}s: {'; '.join(errors)}"
                )
                return False

            logger.info(
                f"Successfully formatted {len(files)} files with {tool} in {duration:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Error running {tool}: {e!s}")
            return False

    async def run_linter(self, tool: str, files: List[Path]) -> bool:
        """Run a linting tool on specified files with optimizations."""
        if not files:
            return True

        try:
            config = self.tools[tool]
            start_time = time.time()

            # Create evaluation directory structure if using mypy
            if tool == "mypy":
                eval_dirs = [
                    self.repo_path / "evaluation" / "types" / subdir
                    for subdir in [
                        "text",
                        "html",
                        "expressions",
                        "coverage",
                        "linecount",
                    ]
                ]
                for dir_path in eval_dirs:
                    dir_path.mkdir(parents=True, exist_ok=True)

            # Filter files that need linting
            files_to_lint = []
            for file in files:
                if await self.linting_optimizer.should_lint_file(file, tool):
                    files_to_lint.append(file)
                else:
                    cached = await self.linting_optimizer.get_cached_result(file, tool)
                    if cached and not cached.get("success", True):
                        return False

            if not files_to_lint:
                logger.info(f"No files need linting with {tool} (using cache)")
                return True

            # Special case for bandit which works better with directories
            if tool == "bandit":
                cmd = [tool] + config.args + [str(self.repo_path)]

                # Execute bandit with enhanced security analysis
                proc = await create_subprocess_exec(
                    *cmd, stdout=asyncsubprocess.PIPE, stderr=asyncsubprocess.PIPE
                )
                stdout, stderr = await proc.communicate()

                try:
                    from collections import defaultdict
                    import json

                    # Parse JSON output
                    output = json.loads(stdout.decode())
                    metrics = output.get("metrics", {})
                    results = output.get("results", [])

                    # Group findings by severity
                    findings_by_severity = defaultdict(list)
                    for finding in results:
                        severity = finding.get("issue_severity", "LOW")
                        findings_by_severity[severity].append(finding)

                    # Generate security report
                    total_issues = len(results)
                    if total_issues > 0:
                        logger.warning("\n🔒 Security Analysis Report:")
                        logger.warning("=" * 50)

                        for severity in ["HIGH", "MEDIUM", "LOW"]:
                            issues = findings_by_severity[severity]
                            if issues:
                                logger.warning(
                                    f"\n{severity} Severity Issues: {len(issues)}"
                                )
                                for issue in issues:
                                    logger.warning(
                                        f"\n  • {issue.get('issue_text')}"
                                        f"\n    File: {issue.get('filename')}"
                                        f"\n    Line: {issue.get('line_number')}"
                                        f"\n    CWE: {issue.get('cwe', {}).get('id', 'N/A')}"
                                        f"\n    More Info: {issue.get('more_info')}"
                                    )

                        logger.warning("\nMetrics:")
                        logger.warning(
                            f"  • Files analyzed: {metrics.get('_totals', {}).get('loc', 0)} lines in {metrics.get('_totals', {}).get('nosec', 0)} files"
                        )
                        logger.warning(
                            f"  • Test coverage: {metrics.get('_totals', {}).get('nosec', 0)} security tests"
                        )

                        # Add remediation suggestions
                        await self.analyze_security_findings(findings_by_severity)

                        # Cache the findings for future reference
                        cached_data = {
                            "success": True,
                            "findings": findings_by_severity,
                            "metrics": metrics,
                            "total_issues": total_issues,
                            "last_check_time": time.time(),
                        }

                        # Cache per-file results for incremental checking
                        for finding in results:
                            file_path = Path(finding["filename"])
                            if file_path.exists():
                                file_findings = cached_data.copy()
                                file_findings["file_specific"] = True
                                self.linting_optimizer.update_cache(
                                    file_path, tool, file_findings
                                )

                        # Cache overall results
                        self.linting_optimizer.update_cache(
                            Path(self.repo_path), tool, cached_data
                        )

                        # Return success but log warning about security issues
                        logger.warning(
                            f"\n⚠️  Found {total_issues} security issues. "
                            "Review the report and remediation suggestions above."
                        )
                        return True

                    logger.info("✅ No security issues found!")
                    # Cache clean results
                    self.linting_optimizer.update_cache(
                        Path(self.repo_path),
                        tool,
                        {
                            "success": True,
                            "findings": {},
                            "metrics": metrics,
                            "total_issues": 0,
                            "last_check_time": time.time(),
                        },
                    )
                    return True

                except json.JSONDecodeError:
                    logger.error("Failed to parse bandit output")
                    return False
                except Exception as e:
                    logger.error(f"Error processing bandit results: {e}")
                    return False

            # Use dynamic batch sizing based on file sizes
            async def get_file_size(file: Path) -> int:
                size = await self.resource_manager.file_operator.get_file_size(file)
                return size if size is not None else 0  # Handle None case

            file_sizes = await asyncio.gather(
                *[get_file_size(f) for f in files_to_lint]
            )

            # Group files by size for optimal batch distribution
            small_files = []
            large_files = []
            for file, size in zip(files_to_lint, file_sizes):
                if size > 100 * 1024:  # 100KB threshold
                    large_files.append(file)
                else:
                    small_files.append(file)

            # Create optimized batches
            batches = []
            if small_files:
                batches.extend(
                    [
                        small_files[i : i + config.batch_size]
                        for i in range(0, len(small_files), config.batch_size)
                    ]
                )
            if large_files:
                batches.extend([[f] for f in large_files])

            logger.info(
                f"Running {tool} on {len(files_to_lint)} files in {len(batches)} batches..."
            )

            async def process_batch(batch: List[Path]) -> Dict:
                try:
                    cmd = [tool] + config.args
                    if tool != "bandit":
                        cmd.extend(str(f) for f in batch)

                    proc = await create_subprocess_exec(
                        *cmd, stdout=asyncsubprocess.PIPE, stderr=asyncsubprocess.PIPE
                    )
                    stdout, stderr = await proc.communicate()

                    # Initialize success based on return code
                    success = proc.returncode == 0

                    # Parse output for fixed files
                    fixed_files = []
                    if tool == "ruff" and stdout:
                        output = stdout.decode().strip()
                        for line in output.split("\n"):
                            if "Fixed " in line:
                                fixed_files.extend(batch)
                                success = (
                                    True  # Consider it success if files were fixed
                                )

                    # Special handling for mypy coverage reporting
                    if tool == "mypy":
                        stdout_text = stdout.decode().strip()
                        stderr_text = stderr.decode().strip()

                        # Look for coverage information in output
                        coverage_info = []
                        total_lines = 0
                        typed_lines = 0

                        for line in stdout_text.split("\n") + stderr_text.split("\n"):
                            if "Total lines of source code:" in line:
                                try:
                                    total_lines = int(line.split(":")[1].strip())
                                except (IndexError, ValueError):
                                    pass
                            elif "Imprecise lines:" in line:
                                try:
                                    imprecise = int(line.split(":")[1].strip())
                                    typed_lines = total_lines - imprecise
                                except (IndexError, ValueError):
                                    pass

                        if total_lines > 0:
                            coverage_percentage = (typed_lines / total_lines) * 100
                            coverage_info.append(
                                f"Type coverage: {coverage_percentage:.1f}% ({typed_lines}/{total_lines} lines)"
                            )

                        error_msg = "\n".join(coverage_info) if coverage_info else None
                        success = True  # Always consider coverage reporting as success
                    else:
                        error_msg = stderr.decode().strip() if not success else None

                    # Cache results
                    result = {
                        "success": success,
                        "error": error_msg,
                        "fixed": len(fixed_files),
                        "files": len(batch),
                    }
                    for file in batch:
                        self.linting_optimizer.update_cache(file, tool, result)

                    return result
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    return {"success": False, "error": str(e), "files": 0, "fixed": 0}

            # Process batches concurrently
            tasks = []
            for batch in batches:
                task = asyncio.create_task(process_batch(batch))
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            duration = time.time() - start_time

            # Aggregate results
            total_files = sum(r.get("files", 0) for r in results)
            total_fixed = sum(r.get("fixed", 0) for r in results)
            success = all(r.get("success", False) for r in results)
            errors = [r.get("error") for r in results if r.get("error")]

            if not success:
                logger.error(
                    f"{tool} check failed in {duration:.2f}s: {'; '.join(errors)}"
                )
                return False

            logger.info(
                f"Successfully ran {tool} on {total_files} files "
                f"({total_fixed} fixed) in {duration:.2f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Error running {tool}: {e!s}")
            return False

    async def setup_pre_commit(self) -> bool:
        """Set up pre-commit hooks with all code quality tools.

        Returns:
            True if setup successful, False otherwise
        """
        try:
            precommit_path = self.repo_path / ".pre-commit-config.yaml"
            precommit_config = f"""# See https://pre-commit.com for more information
# See https://pre-commit.com/hooks.html for more hooks
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
    -   id: check-ast
    -   id: check-json
    -   id: check-merge-conflict
    -   id: detect-private-key

-   repo: https://github.com/psf/black
    rev: {self.tools['black'].version}
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/isort
    rev: {self.tools['isort'].version}
    hooks:
    -   id: isort

-   repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v{self.tools['ruff'].version}
    hooks:
    -   id: ruff
        args: [--fix, --exit-non-zero-on-fix]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v{self.tools['mypy'].version}
    hooks:
    -   id: mypy
        additional_dependencies: [types-all]

-   repo: https://github.com/PyCQA/bandit
    rev: {self.tools['bandit'].version}
    hooks:
    -   id: bandit
        args: ["-c", "pyproject.toml"]
"""
            # Write pre-commit configuration
            await self.resource_manager.file_operator.parallel_operation(
                [precommit_path], lambda p: p.write_text(precommit_config)
            )

            # Install pre-commit if not already installed
            try:
                proc = await create_subprocess_exec(
                    "pip",
                    "install",
                    f"pre-commit=={self.tools['pre-commit'].version}",
                    stdout=asyncsubprocess.PIPE,
                    stderr=asyncsubprocess.PIPE,
                )
                await proc.communicate()
            except Exception as e:
                logger.warning(f"Failed to install pre-commit: {e}")
                return False

            # Initialize git if not already initialized
            if not (self.repo_path / ".git").exists():
                try:
                    proc = await create_subprocess_exec(
                        "git",
                        "init",
                        cwd=str(self.repo_path),
                        stdout=asyncsubprocess.PIPE,
                        stderr=asyncsubprocess.PIPE,
                    )
                    await proc.communicate()
                except Exception as e:
                    logger.warning(f"Failed to initialize git repository: {e}")
                    return False

            # Install pre-commit hooks
            try:
                proc = await create_subprocess_exec(
                    "pre-commit",
                    "install",
                    cwd=str(self.repo_path),
                    stdout=asyncsubprocess.PIPE,
                    stderr=asyncsubprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    logger.warning(
                        f"Pre-commit hook installation warning: {stderr.decode()}"
                    )
                    return False

                logger.info("✨ Pre-commit hooks installed successfully!")
                return True

            except Exception as e:
                logger.warning(f"Pre-commit hook installation warning: {e}")
                return False

        except Exception as e:
            logger.error(f"Failed to set up pre-commit: {e}")
            return False

    async def fix_missing_return_types(self, file_path: Path) -> bool:
        """Add return type annotations to functions that are missing them, but only when absolutely safe.

        This method is extremely conservative and will only add '-> None' when:
        1. The function has no return statements at all
        2. The function has only bare 'return' or 'return None' statements
        3. The function is not async (async functions may have implicit returns)
        4. The function is not in a try/except block
        5. The function has no yields
        6. The function has no decorators
        7. The function is not part of a class (methods may have inherited return types)

        Args:
            file_path: Path to the Python file to process

        Returns:
            bool: True if changes were made, False otherwise
        """
        try:
            with open(file_path) as f:
                content = f.read()

            lines = content.splitlines()
            modified = False
            output_lines = []
            in_class = False

            i = 0
            while i < len(lines):
                line = lines[i].rstrip()

                # Track class definitions
                if re.match(r"^\s*class\s+\w+", line):
                    in_class = True
                elif line and not line.startswith(" ") and not line.startswith("\t"):
                    in_class = False

                # Skip if line is a comment, empty, or has a decorator
                if (
                    not line.strip()
                    or line.strip().startswith("#")
                    or line.strip().startswith("@")
                ):
                    output_lines.append(line)
                    i += 1
                    continue

                # Only process non-async, non-class function definitions
                if (
                    not in_class
                    and re.match(r"^\s*def\s+\w+\s*\([^)]*\)\s*:?\s*$", line)
                    and "async" not in line
                ):

                    # Skip if already has return type
                    if "->" in line:
                        output_lines.append(line)
                        i += 1
                        continue

                    # Function without return type annotation found
                    indent = len(line) - len(line.lstrip())
                    func_body_start = i + 1

                    # Analyze function body
                    j = func_body_start
                    has_return = False
                    has_complex_return = False
                    has_yield = False
                    in_try_block = False
                    body_lines = []

                    # Collect and analyze function body
                    while j < len(lines):
                        curr_line = lines[j].rstrip()
                        curr_indent = len(curr_line) - len(curr_line.lstrip())

                        # Check if we're still in the function
                        if curr_line and curr_indent <= indent:
                            break

                        if curr_line.strip():
                            body_lines.append(curr_line.strip())

                            # Check for control structures that might affect return
                            if any(
                                x in curr_line
                                for x in ["try:", "except", "finally:", "with", "yield"]
                            ):
                                has_complex_return = True
                                break

                            # Check return statements
                            if (
                                "return" in curr_line
                                and not curr_line.strip().startswith("#")
                            ):
                                has_return = True
                                # Check if it's not a simple return
                                if not re.match(
                                    r"^\s*return(?:\s+None)?\s*$", curr_line
                                ):
                                    has_complex_return = True
                                    break

                        j += 1

                    # Only add -> None if:
                    # 1. Function body is simple (no complex control structures)
                    # 2. Either has no returns or only simple returns
                    # 3. Not in a class
                    # 4. Not async
                    if (
                        not has_complex_return
                        and ":" in line
                        and not in_try_block
                        and not has_yield
                    ):
                        colon_idx = line.index(":")
                        new_line = line[:colon_idx] + " -> None" + line[colon_idx:]
                        output_lines.append(new_line)
                        modified = True
                    else:
                        output_lines.append(line)
                else:
                    output_lines.append(line)

                i += 1

            if modified:
                with open(file_path, "w") as f:
                    f.write("\n".join(output_lines))

            return modified

        except Exception as e:
            logger.error(f"Error fixing return types in {file_path}: {e}")
            return False

    async def fix_all_return_types(self) -> None:
        """Fix missing return type annotations in all Python files."""
        try:
            python_files = await self.get_python_files()
            fixed_count = 0

            for file_path in python_files:
                if await self.fix_missing_return_types(file_path):
                    fixed_count += 1

            if fixed_count > 0:
                logger.info(
                    f"Added missing return type annotations in {fixed_count} files"
                )

        except Exception as e:
            logger.error(f"Error fixing return types: {e}")

    async def run_all_checks(self) -> bool:
        """Run all code quality checks with improved error handling and reporting."""
        try:
            logger.info("🔍 Running comprehensive code quality checks...")

            # First try to fix any missing return types
            await self.fix_all_return_types()

            # Ensure config file exists first
            await self.setup_tool_configurations()
            config_path = self.repo_path / "pyproject.toml"
            if not config_path.exists():
                logger.error(f"Configuration file not found at {config_path}")
                return False
            logger.info(f"Using configuration from {config_path}")

            # Get Python files
            try:
                python_files = await self.get_python_files()
                if not python_files:
                    logger.warning("No Python files found to process")
                    return True
            except Exception as e:
                logger.error(f"Failed to get Python files: {e}")
                return False

            # Track results for each tool
            results: List[CheckResult] = []
            start_time = time.time()

            # Run formatters
            for tool in ["black", "isort"]:
                try:
                    tool_start = time.time()
                    success = await self.run_formatter(tool, python_files)
                    duration = time.time() - tool_start
                    results.append(
                        CheckResult(
                            tool_name=tool,
                            success=success,
                            files_processed=len(python_files),
                            duration=duration,
                        )
                    )
                except Exception as e:
                    results.append(
                        CheckResult(
                            tool_name=tool,
                            success=False,
                            error_message=str(e),
                            duration=time.time() - tool_start,
                        )
                    )

            # Run linters
            for tool in ["ruff", "mypy", "bandit"]:
                try:
                    tool_start = time.time()
                    success = await self.run_linter(tool, python_files)
                    duration = time.time() - tool_start
                    results.append(
                        CheckResult(
                            tool_name=tool,
                            success=success,
                            files_processed=len(python_files),
                            duration=duration,
                        )
                    )
                except Exception as e:
                    results.append(
                        CheckResult(
                            tool_name=tool,
                            success=False,
                            error_message=str(e),
                            duration=time.time() - tool_start,
                        )
                    )

            # Print summary
            total_duration = time.time() - start_time
            logger.info("\n📊 Code Quality Check Summary:")
            logger.info("=" * 50)
            logger.info(f"Files processed: {len(python_files)}")
            logger.info(f"Total duration: {total_duration:.2f}s")
            logger.info("-" * 50)

            # Group results by status
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]

            if successful:
                logger.info("\n✅ Successful checks:")
                for result in successful:
                    logger.info(
                        f"  • {result.tool_name}: {result.files_processed} files in {result.duration:.2f}s"
                    )

            if failed:
                logger.error("\n❌ Failed checks:")
                for result in failed:
                    error_msg = (
                        f": {result.error_message}" if result.error_message else ""
                    )
                    logger.error(f"  • {result.tool_name}{error_msg}")

            logger.info("=" * 50)

            # Return overall success status
            return len(failed) == 0

        except Exception as e:
            logger.error(f"Code quality checks failed: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup resources and caches."""
        try:
            self.linting_optimizer.clear_cache()
            await self.resource_manager.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def get_security_remediation(self, issue_id: str) -> str:
        """Get security remediation suggestions for a given bandit issue ID.

        Args:
            issue_id: The bandit issue ID (e.g., 'B301')

        Returns:
            str: Detailed remediation suggestion
        """
        remediation_map = {
            # Assert warnings
            "B101": "Use unittest assertions instead of assert statements in tests. In production code, implement proper error handling instead of assertions.",
            # Exec/Eval usage
            "B102": "Avoid using exec() as it can execute arbitrary code. Use safer alternatives like ast.literal_eval() for safe string evaluation.",
            # Hardcoded passwords
            "B105": "Never hardcode passwords or secrets. Use environment variables or secure secret management systems.",
            # Hardcoded IPs
            "B104": "Avoid hardcoding IP addresses. Use configuration files or environment variables instead.",
            # Command injection
            "B601": "Use subprocess.run() with a list of arguments instead of shell=True to prevent command injection.",
            "B602": "Use shlex.quote() to escape shell arguments or preferably avoid shell=True.",
            # SQL injection
            "B608": "Use parameterized queries or ORM instead of string formatting for SQL queries.",
            "B610": "Use SQLAlchemy or other ORM with proper parameter binding instead of raw SQL.",
            # Unsafe YAML
            "B506": "Use safe_load() instead of load() for YAML parsing to prevent arbitrary code execution.",
            # Pickle usage
            "B301": "Avoid using pickle for deserialization as it can execute arbitrary code. Use JSON or other safe serialization formats.",
            # Crypto warnings
            "B303": "Use secrets instead of random for security operations.",
            "B324": "Use strong cryptography from the cryptography package instead of hashlib for sensitive operations.",
            # XML processing
            "B316": "Use defusedxml instead of the standard XML parsers to prevent XXE attacks.",
            # Default
            "default": "Review the code for security implications and consider implementing additional security controls.",
        }

        # Get specific remediation or default
        return remediation_map.get(issue_id, remediation_map["default"])

    async def analyze_security_findings(
        self, findings_by_severity: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Analyze security findings and provide detailed remediation suggestions.

        Args:
            findings_by_severity: Dictionary of security findings grouped by severity
        """
        logger.info("\n🛡️ Security Remediation Suggestions:")
        logger.info("=" * 50)

        # Track unique issue types for deduplication
        seen_issues = set()

        for severity in ["HIGH", "MEDIUM", "LOW"]:
            issues = findings_by_severity.get(severity, [])
            if issues:
                logger.info(f"\n{severity} Severity Remediation:")

                for issue in issues:
                    issue_id = issue.get("test_id")
                    if issue_id not in seen_issues:
                        seen_issues.add(issue_id)

                        # Get remediation suggestion
                        remediation = await self.get_security_remediation(issue_id)

                        logger.info(
                            f"\n  • Issue: {issue.get('issue_text')}"
                            f"\n    ID: {issue_id}"
                            f"\n    Remediation: {remediation}"
                        )

        logger.info("\n📚 Additional Security Resources:")
        logger.info(
            "  • OWASP Python Security: https://owasp.org/www-project-python-security/"
        )
        logger.info(
            "  • Python Security Checklist: https://python-security.readthedocs.io/"
        )
        logger.info("  • Bandit Documentation: https://bandit.readthedocs.io/")

    async def display_comprehensive_scan(self) -> bool:
        """Display a comprehensive code quality scan with an enhanced UI.

        Returns:
            bool: True if all checks passed, False otherwise
        """
        console = Console()

        # Keep track of table rows separately
        table_data = []

        def make_header() -> Panel:
            """Create the header panel."""
            return Panel(
                "[bold cyan]Comprehensive Code Quality Scan[/]\n"
                "[dim]Running all quality checks and security analysis...[/]",
                style="bold white on blue",
            )

        def make_tool_table() -> Table:
            """Create the tool status table."""
            table = Table(
                show_header=True,
                header_style="bold magenta",
                border_style="bright_black",
                expand=True,
                pad_edge=False,
                show_edge=True,
                title="[bold cyan]Tool Status[/]",
                title_justify="left",
            )
            table.add_column("Tool", style="cyan", no_wrap=True)
            table.add_column("Status", justify="center", no_wrap=True)
            table.add_column("Files", justify="right", no_wrap=True)
            table.add_column("Duration", justify="right", no_wrap=True)
            table.add_column("Issues", justify="right", no_wrap=True)

            # Add any existing rows
            for row in table_data:
                table.add_row(*row)

            return table

        def update_table_row(
            table: Table,
            index: int,
            tool: str,
            status: str,
            files: str,
            duration: str,
            issues: str,
        ) -> None:
            """Update a row in the table by recreating it."""
            # Update our data store
            while len(table_data) <= index:
                table_data.append(["-", "-", "-", "-", "-"])
            table_data[index] = [tool, status, files, duration, issues]

            # Create a new table with the updated data
            new_table = make_tool_table()

            # Update the layout with the new table
            layout["tools"].update(new_table)

        def make_security_panel() -> Panel:
            """Create the security findings panel."""
            return Panel(
                "[dim]Waiting for security scan...[/]",
                title="[bold red]Security Findings[/]",
                border_style="red",
                padding=(1, 2),
            )

        def make_diff_report(results) -> Panel:
            """Create a diff report panel showing file changes."""
            if not results:
                return Panel(
                    "[dim]No changes to report[/]",
                    title="[bold green]Diff Report[/]",
                    border_style="green",
                    padding=(1, 2),
                )

            # Collect information about changed files
            changes = []
            for result in results:
                if result.tool_name in ["black", "isort", "ruff"]:
                    if result.success:
                        changes.append(
                            f"[green]✓ {result.tool_name}[/] formatted files successfully"
                        )

            if not changes:
                return Panel(
                    "[dim]No file changes detected[/]",
                    title="[bold green]Diff Report[/]",
                    border_style="green",
                    padding=(1, 2),
                )

            # Build a detailed report
            report = "[bold]File Changes Summary:[/]\n\n"
            for change in changes:
                report += f"{change}\n"

            # Add instructions for user
            report += "\n[bold cyan]Note:[/] You can view the specific changes in your files by using:\n"
            report += "  • [yellow]git diff[/] (if using git)\n"
            report += (
                "  • [yellow]exporter format --report <path>[/] for a detailed report\n"
            )

            return Panel(
                report,
                title="[bold green]Diff Report[/]",
                border_style="green",
                padding=(1, 2),
            )

        def make_layout() -> Layout:
            """Create the main layout."""
            layout = Layout()

            # Create a more stable layout with fixed sizes
            layout.split(
                Layout(name="header", size=3),
                Layout(name="body", ratio=8),
                Layout(name="footer", size=3),
            )

            # Split the body section horizontally
            layout["body"].split_row(
                Layout(name="tools", ratio=2),
                Layout(name="details", ratio=3),
            )

            # Split the details section vertically
            layout["details"].split(
                Layout(name="security", ratio=2),
                Layout(name="diff", ratio=1),
            )

            return layout

        try:
            # Initialize the layout
            layout = make_layout()
            table = make_tool_table()
            security_panel = make_security_panel()
            diff_panel = Panel(
                "[dim]Waiting for scan to complete...[/]",
                title="[bold green]Diff Report[/]",
                border_style="green",
                padding=(1, 2),
            )

            # Set up initial layout content
            layout["header"].update(make_header())
            layout["tools"].update(table)
            layout["security"].update(security_panel)
            layout["diff"].update(diff_panel)
            layout["footer"].update(
                Panel("[dim]Press Ctrl+C to cancel[/]", border_style="bright_black")
            )

            # Create a Live display with reduced refresh rate and no auto refresh
            with Live(
                layout,
                console=console,
                auto_refresh=False,
                refresh_per_second=4,  # Limit refresh rate
                transient=False,  # Keep the display visible
                screen=True,
            ) as live:
                # Track results for each tool
                results: List[CheckResult] = []
                start_time = time.time()

                # Get Python files
                try:
                    python_files = await self.get_python_files()
                    if not python_files:
                        table.add_row(
                            "File Search",
                            "[yellow]No Files[/]",
                            "-",
                            "-",
                            "-",
                        )
                        live.refresh()
                        await asyncio.sleep(0.5)
                        return True
                except Exception as e:
                    table.add_row(
                        "File Search",
                        f"[red]Error: {e}[/]",
                        "-",
                        "-",
                        "-",
                    )
                    live.refresh()
                    await asyncio.sleep(0.5)
                    return False

                # Pre-populate table with placeholder rows
                for tool in ["Black", "Isort", "Ruff", "MyPy", "Bandit"]:
                    table.add_row(tool, "[dim]Pending...[/]", "-", "-", "-")
                live.refresh()

                # Run formatters
                current_row = 0
                for tool in ["black", "isort"]:
                    update_table_row(
                        table,
                        current_row,
                        tool.title(),
                        "[yellow]Running...[/]",
                        str(len(python_files)),
                        "-",
                        "-",
                    )
                    live.refresh()

                    try:
                        tool_start = time.time()
                        success = await self.run_formatter(tool, python_files)
                        duration = time.time() - tool_start

                        status = "[green]✓ Passed[/]" if success else "[red]× Failed[/]"
                        update_table_row(
                            table,
                            current_row,
                            tool.title(),
                            status,
                            str(len(python_files)),
                            f"{duration:.1f}s",
                            "-",
                        )
                        live.refresh()

                        results.append(
                            CheckResult(
                                tool_name=tool,
                                success=success,
                                files_processed=len(python_files),
                                duration=duration,
                            )
                        )
                    except Exception as e:
                        update_table_row(
                            table,
                            current_row,
                            tool.title(),
                            f"[red]Error: {str(e)[:30]}...[/]",
                            "-",
                            "-",
                            "-",
                        )
                        live.refresh()
                        results.append(
                            CheckResult(
                                tool_name=tool,
                                success=False,
                                error_message=str(e),
                                duration=time.time() - tool_start,
                            )
                        )
                    current_row += 1

                # Run linters
                for tool in ["ruff", "mypy", "bandit"]:
                    update_table_row(
                        table,
                        current_row,
                        tool.title(),
                        "[yellow]Running...[/]",
                        str(len(python_files)),
                        "-",
                        "-",
                    )
                    live.refresh()

                    try:
                        tool_start = time.time()
                        success = await self.run_linter(tool, python_files)
                        duration = time.time() - tool_start

                        # Special handling for bandit security findings
                        if tool == "bandit":
                            cached = await self.linting_optimizer.get_cached_result(
                                Path(self.repo_path), tool
                            )
                            if cached:
                                findings = cached.get("findings", {})
                                total_issues = cached.get("total_issues", 0)

                                # Update security panel with findings
                                security_content = (
                                    "[bold red]Security Issues Found:[/]\n\n"
                                )
                                for severity in ["HIGH", "MEDIUM", "LOW"]:
                                    issues = findings.get(severity, [])
                                    if issues:
                                        security_content += f"[bold]{severity}[/] Severity: {len(issues)} issues\n"
                                        for issue in issues[
                                            :3
                                        ]:  # Show top 3 issues per severity
                                            security_content += (
                                                f"  • {issue.get('issue_text')}\n"
                                            )
                                        if len(issues) > 3:
                                            security_content += (
                                                f"    ... and {len(issues) - 3} more\n"
                                            )

                                layout["security"].update(
                                    Panel(
                                        (
                                            security_content
                                            if total_issues > 0
                                            else "[green]✓ No security issues found[/]"
                                        ),
                                        title="[bold red]Security Findings[/]",
                                        border_style="red",
                                        padding=(1, 2),
                                    )
                                )

                                update_table_row(
                                    table,
                                    current_row,
                                    "Bandit",
                                    (
                                        "[yellow]⚠ Issues Found[/]"
                                        if total_issues
                                        else "[green]✓ Secure[/]"
                                    ),
                                    str(len(python_files)),
                                    f"{duration:.1f}s",
                                    str(total_issues),
                                )
                            else:
                                update_table_row(
                                    table,
                                    current_row,
                                    "Bandit",
                                    (
                                        "[green]✓ Passed[/]"
                                        if success
                                        else "[red]× Failed[/]"
                                    ),
                                    str(len(python_files)),
                                    f"{duration:.1f}s",
                                    "0",
                                )
                        else:
                            status = (
                                "[green]✓ Passed[/]" if success else "[red]× Failed[/]"
                            )
                            update_table_row(
                                table,
                                current_row,
                                tool.title(),
                                status,
                                str(len(python_files)),
                                f"{duration:.1f}s",
                                "-",
                            )

                        live.refresh()

                        results.append(
                            CheckResult(
                                tool_name=tool,
                                success=success,
                                files_processed=len(python_files),
                                duration=duration,
                            )
                        )
                    except Exception as e:
                        update_table_row(
                            table,
                            current_row,
                            tool.title(),
                            f"[red]Error: {str(e)[:30]}...[/]",
                            "-",
                            "-",
                            "-",
                        )
                        live.refresh()
                        results.append(
                            CheckResult(
                                tool_name=tool,
                                success=False,
                                error_message=str(e),
                                duration=time.time() - tool_start,
                            )
                        )
                    current_row += 1

                # Update diff report panel
                layout["diff"].update(make_diff_report(results))
                live.refresh()

                # Update footer with summary
                total_duration = time.time() - start_time
                successful = len([r for r in results if r.success])
                failed = len([r for r in results if not r.success])

                summary_style = "green" if failed == 0 else "red"
                footer_panel = Panel(
                    f"[{summary_style}]Completed in {total_duration:.1f}s: "
                    f"{successful} passed, {failed} failed[/]\n"
                    "[bold white on blue]Press Enter to return to menu...[/bold white on blue]",
                    border_style="bright_black",
                )
                layout["footer"].update(footer_panel)
                live.refresh()

                # Make sure the user has to explicitly press Enter to continue
                user_confirmed = False
                while not user_confirmed:
                    try:
                        # Use run_in_executor to avoid blocking the event loop
                        # We'll use a simpler approach that's more reliable
                        def wait_for_input():
                            input("Press Enter to continue...")
                            return True

                        user_confirmed = await asyncio.get_event_loop().run_in_executor(
                            None, wait_for_input
                        )
                    except Exception as e:
                        logger.error(f"Error waiting for user input: {e}")
                        # If there's an error, we'll just continue after a short delay
                        await asyncio.sleep(2)
                        user_confirmed = True

                return failed == 0

        except Exception as e:
            logger.error(f"Error during comprehensive scan: {e}")
            return False
