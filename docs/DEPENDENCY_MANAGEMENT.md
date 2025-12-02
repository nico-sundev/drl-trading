# Dependency Management Guide

This document explains how dependencies are managed in the AI Trading workspace, particularly for strategy packages that can be sourced from either the local workspace or external GitLab artifactory.

## Overview

The workspace uses `uv` as the package manager with a centralized configuration. Most dependencies are resolved from the workspace for development consistency, but strategy packages support flexible sourcing.

## Strategy Package Integration

The `drl-trading-strategy-example` package can be sourced from:

- **Local workspace** (default): Used for development and fresh clones
- **GitLab artifactory**: Used for proprietary deployments

### Configuration

In `uv.toml` (unstaged, global config):

```toml
[[index]]
name = "gitlab-artifactory"
url = "https://your-gitlab-instance.com/api/v4/projects/your-project-id/packages/pypi/simple"
explicit = true
```

In `drl-trading-preprocess/pyproject.toml` and `drl-trading-training/pyproject.toml`:

```toml
dependencies = [
    "drl-trading-strategy-example>=0.1.0",  # Version constraint for external resolution
]

[tool.uv.sources]
drl-trading-strategy-example = { path = "../drl-trading-strategy-example" }  # Default to local path
```

Note: The index is defined globally in `uv.toml` for unstaged configuration, similar to Maven's settings.xml. The `.env` override controls which source/index to use.

### Switching to GitLab Artifactory

To use the proprietary version from GitLab artifactory, uncomment the lines in your local `.env` file:

```bash
# In .env:
UV_SOURCES__DRL_TRADING_STRATEGY_EXAMPLE__INDEX="gitlab-artifactory"
UV_INDEX_GITLAB_ARTIFACTORY_USERNAME=your_username
UV_INDEX_GITLAB_ARTIFACTORY_PASSWORD=your_deploy_token
```

Then sync dependencies:

```bash
source .env && uv sync --group dev-full
```

Or use the provided script:

```bash
./sync.sh --group dev-full
```

The script sources `.env` and runs `uv sync` with any arguments passed.

Remove or comment out the lines in `.env` to revert to the local workspace source.

### Behavior

- **Default (no `.env` override)**: Uses local `../drl-trading-strategy-example` path
- **With `.env` (uncommented)**: Overrides to use GitLab artifactory index for version resolution
- **Version resolution**: Resolves based on `>=0.1.0` constraint; if artifactory lacks a matching version, sync may fail (no fallback)

### For CI/CD

In GitLab CI or other deployment pipelines, set the environment variables directly:

```bash
export UV_SOURCES__DRL_TRADING_STRATEGY_EXAMPLE__INDEX="gitlab-artifactory"
export UV_INDEX_GITLAB_ARTIFACTORY_USERNAME="your_username"
export UV_INDEX_GITLAB_ARTIFACTORY_PASSWORD="your_deploy_token"
uv sync
```

### Troubleshooting

- If sync fails with version conflicts, check that the local strategy-example version meets the constraint
- For artifactory access issues, verify the index URL and authentication
- Clear uv cache if needed: `uv cache clean`
