# Contributing to criticality-torch

Thank you for your interest in contributing to criticality-torch!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/willzywiec/criticality.git
cd criticality/criticality-torch
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Running Tests

Run the test suite:
```bash
pytest tests/
```

With coverage:
```bash
pytest --cov=criticality_torch tests/
```

## Code Style

We use:
- **black** for code formatting
- **flake8** for linting
- **mypy** for type checking

Format code before committing:
```bash
black criticality_torch/
flake8 criticality_torch/
mypy criticality_torch/
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests

3. Ensure all tests pass and code is formatted

4. Update documentation if needed

5. Submit a pull request with a clear description

## Reporting Issues

When reporting bugs, please include:
- Python version
- PyTorch version
- Operating system
- Minimal code to reproduce the issue
- Full error traceback

## Questions

For questions about usage or development, please open an issue on GitHub.
