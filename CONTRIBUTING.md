# Contributing to Cognithor

Thank you for your interest in contributing to Cognithor! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Contributor License Agreement (CLA)

By submitting a pull request, you agree to the following:

1. **You own the rights** to the code you are contributing, or have permission from the rights holder.
2. **You grant a perpetual, worldwide, non-exclusive, royalty-free license** to your contributions under the same Apache 2.0 license as the project.
3. **You understand** that your contributions are public and that a record of the contribution (including your name and email) is maintained indefinitely.
4. **You confirm** that your contribution does not contain code from incompatible licenses (GPL, AGPL, proprietary) unless explicitly discussed and approved by a maintainer.

No separate CLA document needs to be signed. Submitting a PR constitutes acceptance of these terms. If your employer has intellectual property claims on your work, please ensure you have permission before contributing.

## How to Contribute

### Reporting Bugs

1. **Search existing issues** first to avoid duplicates.
2. Use the **bug report template** (if available) or include:
   - Steps to reproduce
   - Expected vs. actual behavior
   - Python version, OS, and Cognithor version
   - Relevant log output (sanitize any credentials!)
3. Label with `bug`.

### Suggesting Features

1. Open an issue with the `enhancement` label.
2. Describe the use case, not just the solution.
3. Be open to discussion — the maintainers may suggest alternative approaches.

### Submitting Code

#### Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/cognithor.git
cd cognithor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install with dev dependencies
pip install -e ".[all,dev]"

# Verify setup
python -m pytest tests/ -x -q
```

#### Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Write code** following the project conventions (see below).
3. **Write tests** for new functionality.
4. **Run the full check suite:**
   ```bash
   python -m pytest tests/ -x -q      # All tests pass
   ruff check src/ tests/              # No lint errors
   ruff format --check src/ tests/     # Formatting correct
   mypy src/jarvis/ --strict           # Type checking passes
   ```
5. **Commit** with a clear message:
   ```
   Add Grok-4 model defaults to xAI provider config
   ```
6. **Push** and open a Pull Request against `main`.

#### Code Conventions

- **Python 3.12+** — use modern syntax (`match/case`, `type` aliases, `X | Y` unions)
- **Type hints everywhere** — all functions, all parameters, all return types
- **Pydantic models** for data structures
- **Async-first** — use `async`/`await` for I/O operations
- **No print()** in library code — use `structlog` for logging
- **Line length:** 100 characters (configured in `pyproject.toml`)
- **Imports:** sorted by `ruff` (isort-compatible)
- **Docstrings:** in English, for classes and public functions
- **Security:** never log credentials, always sanitize user input at system boundaries

#### Test Conventions

- **Test file naming:** `tests/test_{module}/test_{file}.py`
- **Test class naming:** `TestClassName` grouping related tests
- **Test method naming:** `test_specific_behavior_under_condition`
- **Use fixtures** from `conftest.py` (`tmp_jarvis_home`, `config`, etc.)
- **Mock external services** — tests must run without network access
- **Target:** maintain 89%+ coverage, 0 test failures

## Review Process

### For Contributors

1. All PRs require **at least one maintainer review** before merging.
2. PRs must pass **all CI checks** (tests, lint, type-check).
3. Keep PRs focused — one feature or fix per PR.
4. Respond to review feedback promptly.
5. Squash commits if requested by a reviewer.

### For Reviewers

1. Review within **48 hours** or assign to another maintainer.
2. Be constructive and specific in feedback.
3. Check for:
   - Correctness and completeness
   - Test coverage for new code
   - Security implications (especially for MCP tools, shell, filesystem)
   - Performance impact
   - Consistency with existing patterns
4. Approve or request changes — avoid "nit-only" blocks on otherwise good PRs.

### Merge Policy

- **Maintainers merge** after approval — contributors should not merge their own PRs.
- **Squash merge** is preferred for feature branches.
- **Rebase merge** for clean, atomic commits that each pass tests.

## Maintainer Guidelines

### Current Maintainers

| Name | GitHub | Role |
|------|--------|------|
| Alexander Soellner | @cognithor | Project Lead, Final Approval |

### Maintainer Responsibilities

1. **Triage issues** within 1 week of submission.
2. **Review PRs** within 48 hours or delegate.
3. **Maintain CI/CD** — all checks must be green on `main`.
4. **Release management** — follow semantic versioning (SemVer).
5. **Security response** — critical vulnerabilities patched within 24 hours.
6. **Decision authority** — maintainers have final say on architectural decisions.

### Becoming a Maintainer

Consistent, high-quality contributions over time may lead to maintainer status. The project lead decides when to invite new maintainers based on:

- Quality and frequency of contributions
- Understanding of the codebase architecture
- Responsiveness to reviews and community interactions
- Alignment with the project's design principles

## Architecture Principles

When contributing, keep these core principles in mind:

1. **PGE Trinity is sacred** — Planner (LLM), Gatekeeper (deterministic), Executor (sandboxed). Never bypass the Gatekeeper.
2. **Security-first** — every tool call goes through policy validation. No exceptions.
3. **Local-first** — cloud providers are optional. Core functionality must work with Ollama alone.
4. **No magic** — explicit configuration over convention. Users should understand what the system does.
5. **Test everything** — if it's not tested, it doesn't exist.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

---

Questions? Open an issue with the `question` label or reach out to the maintainers.
