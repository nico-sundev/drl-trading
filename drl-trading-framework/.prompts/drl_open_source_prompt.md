
# DRL Trading Project Open Source Strategy

I'm working on a Deep Reinforcement Learning (DRL) trading project hosted on **GitLab**, which includes both private logic and a reusable framework. I want to **open-source only the framework part**, keeping the trading strategy, environment, config, and features private.

---

## Current Project Structure (Private GitLab Repo)

Contains everything:
- Click-based terminal CLI for app start
- Centralized config defining all parameters and features
- Abstract base classes for feature engineering
- A full preprocessing pipeline (from raw data to train-ready format)
- Training/validation/test spin-up
- Upcoming: backtesting (BT library) and inference modules

---

## Goal: Two-Repos Architecture

### 1. Public GitLab repo (mirrored to GitHub)
- **Name**: `drl-framework`
- **Contains**:
  - Preprocessing pipeline (generic parts)
  - Abstract base classes
  - Utility functions
  - Training spin-up logic (but no custom environment or configs)
  - CLI scaffolding
- **Structured as a Python package**
- **Excludes**:
  - Project-specific configs
  - Strategies or datasets

### 2. Private GitLab repo
- **Name**: `drl-trading-private`
- **Contains**:
  - Custom features implementing abstract base classes
  - Project-specific configs
  - Custom environment and agent logic
  - Uses `drl-framework` as a dependency (via GitLab path or submodule)

---

## Requirements for the AI Agent

### 1. Code/Structure Tasks
- Generate a clean Python package layout for `drl-framework`
- Provide reusable, minimal examples for:
  - Preprocessing pipeline class
  - Abstract base class for features
  - CLI scaffold using `click`
  - Training orchestrator entry point
- Set up GitLab CI template for testing and building the public repo
- Optionally, show how to mirror to GitHub with minimal config

### 2. Integration in Private Repo
- Show how to import the `drl-framework` as a GitLab submodule or editable pip dependency
- Provide an example feature that implements a public abstract base class
- Demonstrate use of the framework's CLI and training scaffold within private logic

### 3. Documentation/README
- Generate a sample `README.md` for the public repo
- Include usage example, minimal install/setup guide, and contribution instructions

### 4. Versioning/Compatibility Suggestions
- Provide guidance for how to handle versioning between public/private repos
- Recommend testing strategy to ensure compatibility between framework and private logic

---

## Constraints
- Framework code must be generic and safe to open source
- No leaking of proprietary or model-specific configurations
- Public repo must stand alone with documentation and basic functionality to attract contributors
- Must be maintainable long-term (e.g., CI/CD, tests, modularity)
