# Project Structure Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository around stable capability domains so the existing retrieval work and upcoming QA / agent assignments can live in one maintainable codebase without task-number folder names.

**Architecture:** Keep a single installable Python package under `src/ecom_qa/`, collapse the code into a small set of broad domains (`data`, `retrieval`, `agent`, `evaluation`, `common`), move dataset assets into reusable subfolders, and simplify the CLI layer to a few durable script entry points. Preserve generated reports and run artifacts as historical outputs while updating active docs and code to the new layout.

**Tech Stack:** Python 3.10+, uv, pytest, Hatchling

---

### Task 1: Add regression coverage for the new layout

**Files:**
- Create: `tests/test_layout.py`
- Modify: `pyproject.toml`

- [ ] Add a smoke test that imports the new top-level package and verifies key modules resolve.
- [ ] Add a filesystem test that checks the new dataset folders and renamed scripts exist.
- [ ] Add `pytest` as a dev dependency so the smoke tests can run locally.

### Task 2: Reorganize source and dataset layout

**Files:**
- Create: `src/ecom_qa/`
- Modify: `src/ecom_rag/*`
- Modify: `dataset/*`
- Modify: `scripts/*`

- [ ] Rename the package from `ecom_rag` to `ecom_qa`.
- [ ] Move source files into the broader domain folders:
  - `data/`
  - `retrieval/`
  - `agent/`
  - `evaluation/`
  - `common/`
- [ ] Rename the active scripts to stable, assignment-agnostic names.
- [ ] Move dataset metadata files into `dataset/catalog/` and `dataset/manifests/`.

### Task 3: Repair imports and path references

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `reports/*.md`
- Modify: `src/ecom_qa/**/*`
- Modify: `scripts/*`

- [ ] Update all Python imports to the new package paths.
- [ ] Update script defaults and any hard-coded paths that refer to moved files.
- [ ] Refresh the README layout, quick start, and script examples.
- [ ] Update non-generated reports and notes that reference old source or script paths.

### Task 4: Verify the restructure

**Files:**
- No code changes expected

- [ ] Run `pytest` for the new smoke tests.
- [ ] Run `python <script> --help` for each renamed script.
- [ ] Run a package import smoke check to confirm the renamed package installs cleanly from `src/`.
