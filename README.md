# Science AI

A fully local, chat-first research system. One local model (served by Ollama) decides what to do;
deterministic tools (SymPy, NumPy/SciPy, mpmath, pint) do every calculation; a graph records every
step, its status and the evidence that verified it.

Phase 1 covers mathematics. Phase 2 adds physics and engineering: values with units, physical
constants (CODATA 2022 from the pinned SciPy), ODEs (closed form and numeric, each checking the
other), linear systems and DC circuits, with modelling assumptions you confirm or reject. Phase 3
adds data: import CSV, TSV and Excel files, then describe, filter, group, test
(t-tests, Mann-Whitney, ANOVA, Kruskal-Wallis, chi-square, correlation) and fit (least squares),
with effect sizes, confidence intervals, assumption diagnostics, Holm correction when several tests
run on the same data, and plots. Phase 4 (this branch) adds computational chemistry for drug
discovery: structure identity, descriptors, drug-likeness filters, similarity and clustering,
substructure search, candidate ranking and a local literature search. Design:
[docs/design.md](docs/design.md).

Opening an older knowledge store upgrades it in place, one schema version at a time (v1 to v2 to
v3 to v4). A backup is written next to it first (`knowledge.db.v1.bak`, `knowledge.db.v2.bak`); if
an upgrade fails, that step is rolled back and the store keeps working at its previous version.

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

## Working with data

Add a file with the **+** button or **Files > Add file...**, or drop it anywhere on the window.
`.csv`, `.tsv`, `.tab`, `.xlsx` and `.xlsm` files are imported: the file is hashed and a copy is
kept in `$HOME\.sciai\datasets` (so editing or deleting the original can't change an analysis),
then it is read twice, by pandas and by a second reader (Python's csv module or openpyxl), and the
import is refused unless both reads agree. Other files are copied to `$HOME\.sciai\files`.

Then ask about the file by name: "In trial.csv, is the score different between groups A and B?"

- The model sees the column names, types, units, missing counts and the levels of small categorical
  columns. It never sees row values; every number comes from a tool.
- Each test reports its statistic, p value, effect size with a confidence interval, and alpha
  (0.05, set in `[data] alpha`, or a different one named in the question). It is checked
  independently (a direct recomputation, then a permutation test or statsmodels).
- The test's assumptions (normality, equal variances, expected counts, independence) become
  assumption nodes with their diagnostics. One the data speak against is marked doubtful, the
  answer carries a caution, and the model is told the robust alternative.
- Several tests on the same file are a family: the answer reports Holm-adjusted p values too.
- Importing the same bytes again reuses everything computed on them, with no model call. A
  changed cell is a new dataset, and answers about the old one are not reused.
- **Files** lists the imported datasets (rows x columns). **Remove** deletes one and invalidates
  every result, plot and answer built on it, in every session.
- In the workspace, a dataset (or any step that read one) has a **Data** tab with the schema and
  the first 50 rows; a plot has a **Plot** tab with PNG and SVG export.

`[data] max_rows` (5,000,000) and `[data] max_file_mb` (400) refuse files too large to load safely.

## Working with molecules

Ask about a structure by writing it as SMILES, a molblock or an InChI: "What are the molecular
weight and TPSA of `Cn1cnc2c1c(=O)n(C)c(=O)n2C`?". Every structure is standardized on the way in
(largest fragment, neutralized, canonical tautomer where one is defined) and read twice: the
canonical SMILES is parsed again from scratch and the import is refused unless the InChIKey,
formula and atom and bond counts agree both times.

What this part of the system does is screening, and the whole of it is computational:

- **Identity and descriptors**: canonical SMILES, InChI and InChIKey, formula, molecular weight,
  monoisotopic mass, heavy atoms, rings, HBD, HBA, TPSA, rotatable bonds, sp3 fraction.
- **Drug-likeness**: the Lipinski and Veber filters, with the failing terms named. logP is a
  Crippen estimate and is labelled a single-method estimate, because there is no genuinely
  different second method here to compare it against.
- **Similarity and clustering**: Tanimoto over Morgan fingerprints, Butina clusters, SMARTS
  substructure search, and a weighted ranking over stored descriptor values.
- **Literature**: a search over PDFs and text files you put in `$HOME\.sciai\corpus`, returning
  quoted passages with their file, page and offset. Methods and procedure sections are never
  indexed, so they cannot be searched or returned.

Three things hold whatever the question is:

- **Every chemistry result is a hypothesis.** It is never promoted to verified, by any route: the
  store, the graph engine and the verifier each refuse it independently. An answer built on one is
  not verified and says so. Checks still run, and every one of them is required; what passing buys
  is that the result may be reused, still labelled a hypothesis. The **Molecule** tab shows which
  checks passed next to that label.
- **The system declines what it cannot do.** A request is declined when no registered tool
  performs the operation it asks for, which is the whole test. Synthesis routes, reaction
  procedures, compounding instructions and doses are declined because nothing here produces them.
  Keywords only choose which sentence you read; they never decide whether to decline, so asking
  about a compound named in a synthesis paper is answered normally.
- **A restricted-structure screen** runs on every structure that enters, and again on every
  neighbour a library search returns. It refuses chemical-weapon agents and their close analogs,
  and refuses ranking criteria that ask for toxicity or lethality to be maximized. It is a safety
  net, not a guarantee: it is partial by construction, and it is not a substitute for the legal
  and ethical review that real work in this area needs.

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
python scripts\live_check.py --model qwen3:8b --no-think --trace
python scripts\live_check.py --model qwen2.5:7b-instruct-q4_K_M --suite physics
python scripts\live_check.py --model qwen2.5:7b-instruct-q4_K_M --suite data
python scripts\live_check.py --model qwen2.5:7b-instruct-q4_K_M --suite chem
```

It runs six fixed problems (a definite integral, an equation, a unit conversion, an ODE, one hard
multi-step problem that small models usually fail, and a repeat of the first to test knowledge
reuse) on a fresh temporary store, so your own knowledge store is not touched. It prints a table
with pass/fail, JSON retries, the failure-ladder stage reached, seconds and model calls, and saves
it to `results\<model>.md` (a `:` or `/` in the tag becomes `-`, since Windows file names can't
hold them). If Ollama isn't running or the model isn't pulled, it says so and exits with code 2.

`--suite physics` runs six physics problems instead (a projectile in degrees, an absolute
temperature, a photon energy from constants, an RC discharge, a series/parallel circuit, and a
repeat of the first) and saves `results\<model>-physics.md`. Nobody is there to tick the
assumption checklist, so the defaults are accepted and listed in the report.

`--suite data` imports the three files in `tests\fixtures\data` (a CSV, R's PlantGrowth as TSV and
R's sleep as an Excel sheet) and asks five questions about them (a Welch t-test, a one-way ANOVA, a
regression slope, a t-test on the Excel sheet, group means) plus a repeat of the first. Answers are
checked against R's reference values to 4 significant digits. It saves `results\<model>-data.md`.

`--trace` writes every exchange to `results\trace-<suite>-<n>.jsonl`, one file per problem and
one JSON object per model call: the system prompt, the user prompt, the JSON schema the request
sent as the output format, and the raw reply. Read it when a suite fails at formalization, since
that is the only place the prompt, the schema and the reply sit side by side.

`--suite chem` first standardizes five spellings of aspirin with no model call, and reports whether
they collapse to one InChIKey (if they do not, nothing else the suite says about chemistry means
anything). It then asks eight questions: identity, descriptors against reference values, a logP, a
drug-likeness verdict, a similarity ranking over a three-member library, two requests nothing can
serve (a synthesis route and a dose, which must be declined with no tool run at all) and a repeat
to test reuse. A chemistry answer passes only if it comes back a hypothesis; an answer that comes
back verified is a failure. It saves `results\<model>-chem.md`.

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
2. A question asking for something no tool performs (a synthesis route, a dose) is declined
   here, in code, before any model call.
3. One model call routes the question to a recipe and fills that recipe's slots. The recipe list
   is a closed enum, so this is the whole of the model's planning; a question no recipe covers
   comes back as out of scope with the list of what there is. The root node holds the slots, and
   you confirm or edit it. The recipe also names the problem type, so the type's default
   assumptions (no air resistance, ideal wires...) are offered as a checklist; each ticked one
   becomes an assumption node, and rejecting it later invalidates everything built on it.
4. Code then runs the recipe's tool calls, building each from the slots and the results before
   it, so an easy question costs exactly that one model call. A call whose result is already
   verified in the store is reused instead of run.
5. Hard-coded risk rules (stakes, surprise, confidence, step type, units) decide when a node needs
   an independent check; a recipe's own tools are always checked, including that the answer is in
   the unit the question asked for. A failed check walks the ladder: retry with a different method
   or by the specialist role for that tool, backtrack, then escalate with both results side by side.
6. A dataset or molecule question goes to the step-by-step controller loop instead: one JSON action
   per model call, validated, executed in the tool sandbox.
7. The final answer is a template whose values are filled from node results, so the model never
   writes a number. For a recipe the sentence is written by code, and it never names the unit,
   because the value renders with it.

### The recipes

`unit_convert` (including degC and degF), `definite_integral`, `solve_equation`,
`area_between_curves`, `ode_ivp`, `projectile_range`, `photon_energy`, `rc_discharge`,
`series_parallel_current`. Each one has named slots -- numbers, units, expressions, variables --
and `src/sciai/controller/recipes.py` holds them, their two worked examples apiece, and the tool
calls each one makes. A number slot takes only the number; its unit lives in the matching
`*_unit` slot, and an expression slot takes an expression, which is what the old quantity-only
`givens` could not hold.
