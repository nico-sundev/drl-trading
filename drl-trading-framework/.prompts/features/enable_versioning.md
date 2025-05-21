# 🔧 TASK OVERVIEW

You are working on a Python project that uses `pyproject.toml` with Poetry and the `poetry-dynamic-versioning` plugin. The goal is to **automate versioning and publishing** of this project as a GitLab package, and to make it **consumable by another private project via Git tags**.

---

## 🎯 OBJECTIVES

1. **Inspect the project structure** and configuration:
   - Locate and understand `pyproject.toml`
   - Confirm usage of `poetry-dynamic-versioning`
   - Detect dependencies, project name, and existing versioning conventions
   - Check if Git tags are used and follow `vX.Y.Z` format

2. **Enable automated Git-based versioning using GitLab CI**:
   - Create a `.gitlab-ci.yml` pipeline that:
     - Installs dependencies (`poetry`, `poetry-dynamic-versioning`)
     - Builds the Python package on **Git tag push**
     - (Optional) publishes the package to GitLab’s internal PyPI repository

3. **Ensure the project can be referenced by another GitLab repo (impl project)**:
   - Via `pyproject.toml`, the impl project should be able to do:
     ```toml
     [tool.poetry.dependencies]
     drl-framework = { git = "https://gitlab.com/yourgroup/drl-framework.git", tag = "v0.3.0" }
     ```

4. **Add GitLab release support** (optional but preferred):
   - For each tag like `v0.3.0`, a release should be created via GitLab Releases
   - Include auto-generated changelog if possible

---

## 🔍 PROJECT DETAILS

- Uses **Poetry** and **pyproject.toml**
- Uses **poetry-dynamic-versioning** plugin
- Target project name: `drl-framework`
- Git repository hosted on **GitLab**
- Git tags used for semantic versioning (`vX.Y.Z`)
- Separate private GitLab repo (`drl-trading-impl`) will consume this via Git ref

---

## ✅ EXPECTED OUTCOMES

1. A working `.gitlab-ci.yml` that:
   - Runs only on Git tags
   - Installs poetry + versioning plugin
   - Builds the package with correct version
   - Optionally uploads it to GitLab’s PyPI

2. Optional:
   - GitLab Release created from tag
   - Artifacts attached (wheels, dist, etc.)

3. Verified that the `drl-trading-impl` repo can reference the framework repo using:
   ```toml
   drl-framework = { git = "https://gitlab.com/yourgroup/drl-framework.git", tag = "v0.3.0" }
