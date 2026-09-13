# AGENTS.md

## Python environment

- Required Python version: **3.10.8**.
- Never install packages into the system/global Python. Always use a project-local
  virtual environment.
- Setup order, every time dependencies are needed in this repo:
  1. Create the virtual environment first (only if `.venv` does not already exist):
     `C:\Python310\python.exe -m venv .venv`
  2. Only after the virtual environment exists, install/update Python dependencies
     into it:
     `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
- Always invoke Python, pip, pytest, etc. through `.\.venv\Scripts\python.exe`
  (or `.\.venv\Scripts\python.exe -m <tool>`), never through a bare `python`/`pip`
  that might resolve to a different (system or wrong-version) interpreter.
- If `requirements.txt` changes, re-run step 2 to sync the existing `.venv`
  before running or testing any code.
