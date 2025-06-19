# Exporters

A comprehensive code quality and documentation tool suite that integrates with globaltools for enhanced code formatting capabilities.

## Features

- 🔍 **Code Quality Tools**: Black, isort, Ruff, MyPy, Bandit
- 📚 **Documentation Generation**: Sphinx with multiple themes and extensions
- 🛠️ **Multi-language Formatting**: Support for Python, JavaScript, TypeScript, Go, Rust, C/C++, and more
- ⚡ **Performance Optimized**: Async operations with batched processing
- 🎨 **Rich UI**: Interactive terminal interface with progress indicators
- 🔗 **Globaltools Integration**: Automatic tool installation and management

## Installation

### Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### External Tool Dependencies

The exporter integrates with various external formatting tools. Install them as needed:

```bash
# JavaScript/TypeScript tools
npm install -g prettier eslint

# Go tools
go install mvdan.cc/sh/v3/cmd/shfmt@latest

# Shell tools (varies by OS)
# On macOS with Homebrew:
brew install shellcheck

# On Ubuntu/Debian:
apt-get install shellcheck

# JSON processing
# On macOS:
brew install jq
# On Ubuntu/Debian:
apt-get install jq

# Docker linting
# Install from: https://github.com/hadolint/hadolint

# Ruby tools
gem install rubocop

# PHP tools
composer global require friendsofphp/php-cs-fixer
```

## Usage

### Command Line Interface

The exporter can be used in two ways:

#### 1. Interactive UI Mode
```bash
# Run the interactive interface
exporter
```

#### 2. Direct Formatting Mode
```bash
# Format files in current directory (non-recursive)
exporter format . --no-recursive

# Format files recursively
exporter format /path/to/project

# Format a single file
exporter format myfile.py

# Show detailed diff report
exporter format . --report
```

### Python API

```python
import asyncio
from pathlib import Path
from exporters.beautifier import CodeQualityManager
from exporters.documenter import SphinxDocumenter
from internal.resource_manager import get_resource_manager

async def main():
    repo_path = Path(".")
    resource_manager = get_resource_manager(repo_path / "cache")
    
    # Code quality management
    async with CodeQualityManager(repo_path, resource_manager) as quality_manager:
        await quality_manager.setup_tool_configurations()
        await quality_manager.run_all_checks()
    
    # Documentation generation
    documenter = SphinxDocumenter(str(repo_path))
    await documenter.ensure_sphinx_packages()
    await documenter.setup_sphinx_docs()

if __name__ == "__main__":
    asyncio.run(main())
```

## Supported File Types

| Language | Extensions | Formatters |
|----------|------------|------------|
| Python | `.py` | black, isort, ruff |
| JavaScript | `.js`, `.jsx` | prettier, eslint |
| TypeScript | `.ts`, `.tsx` | prettier, eslint |
| JSON | `.json` | prettier, jq |
| Markdown | `.md` | prettier |
| Shell | `.sh`, `.bash`, `.zsh` | shfmt, shellcheck |
| Go | `.go` | gofmt |
| Rust | `.rs` | rustfmt |
| C/C++ | `.c`, `.cpp`, `.h`, `.hpp` | clang-format |
| HTML/CSS | `.html`, `.css`, `.scss` | prettier |
| YAML | `.yaml`, `.yml` | prettier |
| Docker | `Dockerfile` | hadolint |
| XML | `.xml` | xmllint |
| Ruby | `.rb` | rubocop |
| Java | `.java` | google-java-format |
| PHP | `.php` | php-cs-fixer |
| Swift | `.swift` | swiftformat |

## Configuration

The tool uses `pyproject.toml` for configuration. Key sections include:

```toml
[tool.black]
line-length = 88
target-version = ["py39"]

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.ruff]
line-length = 88
target-version = "py39"
select = ["E", "F", "B", "N", "UP", "PL", "RUF", "S", "C", "T", "Q"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
disallow_untyped_defs = true

[tool.bandit]
exclude_dirs = ["tests", "venv", ".venv"]
```

## Features

### Code Quality Management
- Comprehensive code quality checks with detailed reporting
- Pre-commit hook setup and management
- Incremental linting with smart caching
- Security vulnerability scanning with Bandit
- Type checking with MyPy

### Documentation Generation
- Automated Sphinx setup with modern themes
- API documentation generation
- Multi-format output (HTML, PDF, etc.)
- Mermaid diagram support
- Custom templates and styling

### Performance Optimization
- Async/await for non-blocking operations
- Batched file processing for large codebases
- Memory-efficient operations with resource management
- Smart caching for incremental updates
- Parallel processing where applicable

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/GriffinCanCode/Exporters.git
cd Exporters

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run code quality checks
python -m exporters.beautifier
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the full test suite
6. Submit a pull request

## Architecture

The project is structured around several key components:

- **`beautifier.py`**: Core code quality management
- **`documenter.py`**: Sphinx documentation generation
- **`ui.py`**: Interactive terminal interface
- **`exporter`**: Main CLI entry point

### Resource Management

The tool uses a sophisticated resource management system for:
- Memory optimization
- File operation batching  
- Performance monitoring
- Cache management

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, feature requests, or questions:
- Create an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the examples in the `examples/` directory 