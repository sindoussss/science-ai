# Science AI

A fully local, chat-first research system. One local model (served by Ollama) decides what to do;
deterministic tools (SymPy, NumPy/SciPy, mpmath, pint) do every calculation; a graph records every
step, its status and the evidence that verified it.

Phase 1 (this branch) covers mathematics. Design: [docs/design.md](docs/design.md).

## Run

```bash
pip install -e ".[dev]"
ollama pull qwen2.5:7b-instruct-q4_K_M   # or any 7-12B 4-bit model; set it in config
python -m sciai                          # light theme by default; SCIAI_THEME=dark for the draft dark theme
```

Configuration: `config/default.toml`, overridden by `~/.sciai/config.toml`. The knowledge store lives
at `~/.sciai/knowledge.db`.

## Test

```bash
pytest                 # unit, acceptance and UI tests; uses a scripted model, no Ollama needed
pytest -m live         # end-to-end against a running local Ollama
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
