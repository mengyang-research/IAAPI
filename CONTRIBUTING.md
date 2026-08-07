# Contributing

Issues and pull requests are welcome, especially for simulator adapters, PEtab compatibility, domain examples, tests, and documentation.

1. Create a focused branch.
2. Install the development dependencies with `python -m pip install -e ".[dev]"`.
3. Add or update tests for behavior changes.
4. Run `python -m pytest` and `python -m build`.
5. Keep scientific claims tied to reproducible artifacts and report failed models or runs explicitly.

Please do not commit datasets, model checkpoints, credentials, or generated run directories.
