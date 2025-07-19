# Project Configuration Consolidation

This document explains the consolidated configuration approach for the AI Trading workspace.

## Overview

The workspace now uses a centralized configuration system where common tool configurations are defined at the workspace level and inherited by individual projects. This reduces duplication and ensures consistency across all projects.

## Workspace-Level Configuration (`pyproject.toml`)

The root `pyproject.toml` contains:

### Shared Tool Configurations
- **Ruff**: Python linting and formatting with consistent rules
- **MyPy**: Static type checking with common settings
- **Pytest**: Testing framework configuration
- **Coverage**: Code coverage reporting settings
- **Black**: Code formatting (for legacy compatibility)

### Shared Dependency Groups
- `dev-common`: Common development tools (mypy, ruff, ipython)
- `test-common`: Common testing dependencies (pytest, pytest-cov, etc.)
- `messaging-common`: RabbitMQ and messaging tools
- `integration-test`: Testing infrastructure (testcontainers, minio)
- `ml-common`: Machine learning libraries (numpy, pandas, gymnasium, stable-baselines3)

## Project-Level Configuration

Each project's `pyproject.toml` now only contains:

### Required Project-Specific Sections
1. **Build System**: Minimal setuptools backend (uv workspace compatible)
2. **Project Metadata**: Name, version, description, dependencies
3. **UV Sources**: Workspace member references
4. **Optional Dependencies**: Project-specific optional deps

### Minimal Build System
Projects use the **most minimal build-system possible**:
```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
```

Benefits:
- **Minimal dependencies**: Only setuptools, no wheel or other extras
- **uv workspace compatible**: Works with `uv build --all-packages`
- **No warnings**: Modern license format prevents deprecation warnings
- **Auto package discovery**: Setuptools auto-discovers src/ layout
- **Clean config**: No manual setuptools configuration needed

### Project-Specific Overrides
Projects can override workspace defaults where needed:
- **Ruff isort**: Custom `known-first-party` packages
- **Coverage**: Project-specific source paths
- **Additional tool settings**: As needed per project

## Benefits

### 1. Reduced Duplication
- Tool configurations defined once at workspace level
- No need to repeat common settings in each project
- Version bumps apply to all projects automatically

### 2. Consistency
- All projects use identical linting, formatting, and testing rules
- Consistent dependency versions across the workspace
- Standardized build and packaging configuration

### 3. Maintainability
- Changes to tool configurations need to be made in one place
- Easier to upgrade tools and dependencies
- Clear separation between shared and project-specific configuration

### 4. Scalability
- Easy to add new projects with minimal configuration
- Shared dependency groups can be referenced by multiple projects
- Tool configurations automatically apply to new projects

## Project Structure Example

```
ai_trading/
├── pyproject.toml                    # Workspace configuration (shared tools & deps)
├── drl-trading-common/
│   └── pyproject.toml                # Project-specific: metadata + dependencies
├── drl-trading-core/
│   └── pyproject.toml                # Project-specific: metadata + dependencies
├── drl-trading-inference/
│   └── pyproject.toml                # Project-specific: metadata + dependencies
└── ...
```

## Migration Summary

The following configurations were moved from individual projects to workspace level:

### Tool Configurations Consolidated
- **Ruff**: Linting rules, line length, target version, ignore patterns
- **MyPy**: Type checking strictness, cache settings, module overrides
- **Pytest**: Test discovery, logging, markers, common addopts
- **Coverage**: Branch coverage, exclusions, omit patterns
- **Black**: Line length, target version, include patterns

### Dependency Groups Created
- Development tools grouped by function
- Testing infrastructure standardized
- ML/data science libraries centralized
- Messaging components grouped

### Project Configurations Simplified
- Removed duplicate tool sections
- **Minimal build system**: Uses bare setuptools (no wheel dependency)
- **Modern license format**: Uses `license = "MIT"` (SPDX format)
- **No license classifiers**: Removed deprecated license classifiers
- **Automatic package discovery**: No manual setuptools configuration needed
- Consistent Python version requirement (>=3.12)
- **Clean and fast**: Minimal dependencies, maximum compatibility

## Usage

### Running Tools
All tools now work consistently across projects:

```bash
# Linting (inherits workspace ruff config)
ruff check .
ruff format .

# Type checking (inherits workspace mypy config)
mypy src/

# Testing (inherits workspace pytest config)
pytest

# Coverage (uses project-specific source paths)
pytest --cov
```

### Adding New Projects
To add a new project:

1. Create minimal `pyproject.toml` with project metadata
2. Add workspace sources if referencing other projects
3. Define project-specific dependencies
4. Tool configurations are automatically inherited

### Customizing Tools
Projects can override workspace defaults:

```toml
# Override ruff isort settings for project-specific imports
[tool.ruff.lint.isort]
known-first-party = ["my_project", "drl_trading_common"]

# Override coverage source paths
[tool.coverage.run]
source = ["my_project"]
```

## Benefits for Development

1. **Faster Setup**: New projects inherit all tool configurations
2. **Consistent Quality**: All projects follow same standards
3. **Easier Maintenance**: One place to update tool versions
4. **Better DX**: Consistent behavior across all projects
5. **Reduced Config Drift**: Shared configurations prevent divergence
