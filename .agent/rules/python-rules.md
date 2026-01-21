---
trigger: always_on
---

# Comprehensive Python & AntiGravity Standards

## 1. Environment & Package Management (uv)
- **Initialization**: If a project lacks a `pyproject.toml`, the agent must initialize it using `uv init`.
- **Tooling**: Use `uv` for all library management, virtual environments, and code execution.
- **Dependencies**: Add new libraries via `uv add <package>` only.
- **Execution**: Run scripts using `uv run <script_name>.py` or uv `uv run -m python <script_name>` to ensure a consistent environment.
- **Reproducibility**: Ensure `pyproject.toml` and `uv.lock` are updated whenever dependencies change.

## 2. Physical & Architectural Constraints
- **File Length**: Strictly NO file shall exceed **150 lines**. 
- **Function Atomicity**: One function = One logic operation. Max **20 lines** per function.
- **Refactoring**: If a file > 150 lines, convert it to a folder and split logic into sub-modules.
- **Small Functions**: Every function must do one thing. If a function contains "and" in its description, it should probably be split.

## 3. Folder & Workspace Structure
- **Domain-Driven**: Group files by functional domain (e.g., `/tracking`, `/calibration`) rather than generic types like `/utils`.
- **Initialization**: Every logic directory must contain an `__init__.py` to define public exports.
- **Naming**: Folder names should be plural for collections (`/models`) and singular for specific components.

## 4. Naming & Encapsulation
- **Variable/Function**: Use `snake_case`. Functions must start with a verb (e.g., `calculate_height`).
- **Boolean Variables**: Must use prefixes `is_`, `has_`, or `should_`.
- **Classes**: Use `PascalCase` nouns.
- **Encapsulation**: Use a leading underscore `_` for all internal/private functions and variables.

## 5. PEP 8 & Core Formatting
- **Standard**: Follow PEP 8 guidelines.
- **Indentation**: Mandatory 4-space indentation; no tabs.
- **Max Line Length**: 79 characters for readability and side-by-side diffs.
- **Vertical Spacing**: Two blank lines between top-level functions/classes.

## 6. Type Safety & Logic
- **Typing**: Mandatory PEP 484 type hints for all parameters and return values.
- **Validation**: Use **Pydantic** models for complex data structures (especially functions with > 3 related arguments).
- **Error Handling**: Catch specific exceptions; never use bare `except:`.
- **Flow**: Use guard clauses and early returns to keep nesting levels below 3.