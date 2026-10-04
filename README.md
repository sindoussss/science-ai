# Science AI

A fully local, chat-first research system. One local model (served by Ollama) decides what to do;
deterministic tools (SymPy, NumPy/SciPy, mpmath, pint) do every calculation; a graph records every
step, its status and the evidence that verified it.

Phase 1 (this branch) covers mathematics. Design: [docs/design.md](docs/design.md).

## Setup on Windows (PowerShell)

You need Python 3.11 or newer, Ollama, and Git. With `winget` (built into Windows 10/11):

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Ollama.Ollama
winget install -e --id Git.Git
```

Close and reopen PowerShell so the new commands are on your PATH. Then get the code and install it
into a virtual environment:

```powershell
git clone https://github.com/sindoussss/science-ai.git
cd science-ai
git checkout phase-1

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If `Activate.ps1` is blocked ("running scripts is disabled on this system"), allow local scripts for
your user once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Pull a model. Ollama starts in the background after install (tray icon); if `ollama list` says it
can't connect, run `ollama serve` in a second PowerShell window.

```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M   # the default; any 7-12B 4-bit model works
ollama list                              # check it is there
```

Every new PowerShell window needs the environment activated again: `.\.venv\Scripts\Activate.ps1`.

## Run

```powershell
python -m sciai                                  # the model from the config
python -m sciai --model qwen3:8b                 # any pulled Ollama tag, for this run only
python -m sciai --model qwen3:8b --think         # reasoning mode on (off by default)
$env:SCIAI_THEME = "dark"; python -m sciai       # draft dark theme
```

Configuration lives in `config\default.toml`. To change a setting for good, copy the keys you need
into `$HOME\.sciai\config.toml` (for example `C:\Users\<you>\.sciai\config.toml`):

```powershell
New-Item -ItemType Directory -Force "$HOME\.sciai" | Out-Null
Set-Content "$HOME\.sciai\config.toml" "[model]`nname = `"qwen3:8b`"`nthink = false"
```

The knowledge store is `$HOME\.sciai\knowledge.db`.

**Thinking models.** `think = false` (the default) turns reasoning off for models that have it
(qwen3, deepseek-r1). Any `<think>...</think>` text a model still writes is removed before its JSON
action is parsed. If you turn thinking on, also raise `num_predict` (for example to 2048), because
thinking tokens count against it.

## Test

```powershell
pytest                       # unit, acceptance and UI tests; scripted model, no Ollama needed
pytest tests\unit -q         # just the fast unit tests
pytest -m live               # end to end against your running Ollama and configured model
```

The UI tests open real (hidden) windows. On a machine without a display, set
`$env:QT_QPA_PLATFORM = "offscreen"` first.

## Live check against a model

```powershell
python scripts\live_check.py --model qwen2.5:7b-instruct-q4_K_M
python scripts\live_check.py --model qwen3:8b --think
```

It runs six fixed problems (a definite integral, an equation, a unit conversion, an ODE, one hard
multi-step problem that small models usually fail, and a repeat of the first to test knowledge
reuse) on a fresh temporary store, so your own knowledge store is not touched. It prints a table
with pass/fail, JSON retries, the failure-ladder stage reached, seconds and model calls, and saves
it to `results\<model>.md` (a `:` or `/` in the tag becomes `-`, since Windows file names can't
hold them). If Ollama isn't running or the model isn't pulled, it says so and exits with code 2.

## Linux / macOS

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ollama pull qwen2.5:7b-instruct-q4_K_M
python -m sciai            # same options as above
pytest
```

## How a question flows

1. Knowledge-store lookup (no model call). A verified answer to the same question is reused.
2. One model call formalizes the question into the root node, which you confirm or edit.
3. If the formal goal is a single tool call that is already verified, it is reused; otherwise it runs.
4. Otherwise the controller loop: one JSON action per model call, validated, executed in the tool sandbox.
5. Hard-coded risk rules (stakes, surprise, confidence, step type) decide when a node needs an
   independent check. A failed check walks the ladder: retry with a different method, backtrack,
   then escalate with both results side by side.
6. The final answer is a template whose values are filled from node results, so the model never
   writes a number.
