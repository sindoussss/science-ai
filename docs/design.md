# Science AI: Phase 0 proposal (file tree + graph schema)

Status: **approved 2026-10-03 with changes** (see section 5). Phase 1 is implemented on the `phase-1` branch.

Defaults picked where the spec was open: PyQt6 UI, one Ollama model (default `qwen2.5:7b-instruct-q4_K_M`, configurable), SQLite + networkx, `pint` for units, pytest for tests.

---

## 1. Design in one paragraph

The controller is a loop in plain Python. Each step it sends the model a short prompt (role system prompt + compact graph summary + last observation) and gets back exactly one JSON action, validated against a schema (Ollama structured output, then `jsonschema`). Code executes the action: a tool runs in a sandboxed CPU worker process, its output becomes a graph node, the risk rules (pure code) decide whether a check is required and which independent checks are allowed, and the verifier runs the chosen check. The UI only observes the graph engine through events; it never talks to the model directly. Every number the user sees comes from a node `result`, never from model text (see 3.6).

## 2. File tree

```
science-ai/
├── pyproject.toml                 # deps pinned; entry point `sciai`
├── README.md
├── config/
│   └── default.toml               # model name, step cap, timeouts, risk thresholds, db path
├── src/sciai/
│   ├── __main__.py                # python -m sciai -> launches UI
│   ├── config.py                  # typed config loader
│   │
│   ├── llm/                       # the ONE model, many roles
│   │   ├── client.py              # Ollama HTTP client, localhost only, keep_alive, token caps
│   │   ├── roles.py               # Role = system prompt + allowed tool names + max tokens
│   │   ├── prompts/               # controller.md, algebra.md, numeric.md, plotting.md (short)
│   │   └── actions.py             # JSON schemas for actions; parse -> validate -> 1 retry -> error
│   │
│   ├── controller/
│   │   ├── loop.py                # reason -> act -> observe -> decide; step cap; stop
│   │   ├── context.py             # builds the compact prompt (graph digest, not full history)
│   │   ├── lookup.py              # knowledge-store lookup BEFORE any model call
│   │   ├── answer.py              # renders final answer from node refs (no model-typed numbers)
│   │   └── steering.py            # chat steer messages + pinned comments -> controller instructions
│   │
│   ├── tools/
│   │   ├── registry.py            # ToolSpec: name, json_schema, domain, kind=solver|checker, fn, cost
│   │   ├── sandbox.py             # process pool, per-call timeout, kill on hang, no network
│   │   ├── parsing.py             # safe SymPy parsing (no eval), canonical forms
│   │   ├── sympy_tools.py         # simplify, expand, factor, solve, diff, integrate, limit, series, equals
│   │   ├── numeric_tools.py       # evaluate, quad, root-find, ODE (NumPy/SciPy)
│   │   ├── units_tools.py         # dimension check, unit conversion (pint)
│   │   ├── plot_tools.py          # numeric node -> PlotSpec (data, not pixels)
│   │   └── data_tools.py          # pandas (Phase 3; stub in Phase 1)
│   │
│   ├── graph/
│   │   ├── model.py               # Node, Edge, enums (Status, Layer, NodeType, EdgeKind)
│   │   ├── engine.py              # networkx DiGraph wrapper; the only writer of node state
│   │   ├── cascade.py             # invalidation propagation (session + cross-session)
│   │   ├── versions.py            # lineage/version bookkeeping for retries and backtracks
│   │   ├── fingerprint.py         # canonical hash used for reuse lookup
│   │   ├── digest.py              # small text summary of the graph for the controller prompt
│   │   └── events.py              # plain-Python observer events (UI bridges them to Qt signals)
│   │
│   ├── verify/
│   │   ├── risk_rules.py          # stakes, surprise, confidence, step-type -> RiskReport
│   │   ├── checks.py              # independent methods: symbolic vs numeric, alt algorithm, known value
│   │   ├── sanity.py              # units, magnitude, sign, domain checks
│   │   ├── verifier.py            # runs checks, writes Evidence, sets verified/failed
│   │   └── ladder.py              # retry (different method/role) -> backtrack -> escalate
│   │
│   ├── store/
│   │   ├── db.py                  # SQLite connection, WAL, migrations
│   │   ├── schema.sql             # tables in section 3.4
│   │   └── repository.py          # load/save, trust rules, reuse queries, cross-session cascade
│   │
│   ├── domains/                   # specialists, added per phase
│   │   ├── math/                  # Phase 1: role prompts + tool set binding
│   │   ├── physics/               # Phase 2
│   │   ├── data/                  # Phase 3
│   │   └── chem/                  # Phase 4: safety.py guard lives here, enforced in engine too
│   │
│   └── ui/
│       ├── theme/
│       │   ├── tokens_light.json  # every color, radius, font, spacing
│       │   ├── tokens_dark.json   # Phase-1 placeholder, filled after light is polished
│       │   └── theme.py           # tokens -> QSS + QPalette
│       ├── main_window.py         # three-pane splitter
│       ├── sidebar.py             # New, sessions, Files, Knowledge base
│       ├── canvas/
│       │   ├── graph_view.py      # QGraphicsView, pan/zoom, live updates
│       │   ├── node_item.py       # status color + label/icon, dashed outline for invalidated
│       │   ├── edge_item.py
│       │   ├── layout.py          # layered DAG layout (depth columns), incremental
│       │   ├── plot_item.py       # matplotlib (Agg) thumbnail inside a node
│       │   ├── pins.py            # numbered pinned comments with text box + Send
│       │   └── legend.py
│       ├── chat_bar.py            # "Ask or steer the controller"
│       ├── inspector/
│       │   ├── inspector.py       # header, version switcher, badge, depends-on chips, Download script
│       │   ├── code_tab.py
│       │   ├── log_tab.py
│       │   ├── chat_tab.py
│       │   ├── env_tab.py         # model, VRAM (nvidia-smi if present), tools
│       │   └── review_tab.py      # evidence, fired rules, ladder position
│       └── controller_thread.py   # runs the loop in a QThread; UI stays responsive
│
└── tests/
    ├── fakes/fake_llm.py          # scripted model for deterministic tests (no Ollama needed)
    ├── unit/                      # one file per module: tools, risk rules, cascade, ladder, store...
    ├── acceptance/                # the five Phase 1 acceptance tests (fake LLM)
    ├── live/                      # same scenarios against real Ollama, marked @live, run manually
    └── ui/                        # pytest-qt smoke tests (status styling, inspector, pins)
```

## 3. Graph and node schema

### 3.1 Keeping the three roles distinct

Every node has a `layer`, and every edge has a `kind`. Rules are written against them, so the roles never blur:

| Layer | What it holds | Edge kinds it uses | Cascades? |
|---|---|---|---|
| `reasoning` | problem, claims, steps, tool results, checks, final answer | `depends_on`, `checks`, `conflicts_with` | yes, along `depends_on` only |
| `domain` | the system under study (a proof structure, a circuit, a dataset's columns) | `relates` (typed via `meta.relation`) | no; tools read it, results land in `reasoning` |
| `plot` | renderable views of numeric data | `renders` (plot -> data node) | plot is invalidated when its data node is |

### 3.2 Node

Each version of a node is its own row. Versions of the same logical node share a `lineage_id`. The canvas shows the latest version; the inspector's v1, v2, ... switcher walks the lineage.

| Field | Type | Notes |
|---|---|---|
| `id` | text (ULID) | identifies this version |
| `lineage_id` | text | stable across retries/backtracks |
| `version` | int | 1, 2, ... |
| `session_id` | text | session that created it |
| `layer` | enum | `reasoning`, `domain`, `plot` |
| `type` | enum | reasoning: `problem`, `claim`, `step`, `tool_result`, `check`, `final`, `error`, `hint`; domain: `entity`; plot: `plot` |
| `domain` | enum | `math`, `physics`, `data`, `chem` |
| `title` | text | short label on the canvas |
| `content` | text | human statement (plain text / LaTeX) |
| `content_canonical` | text | SymPy `srepr` or canonical JSON; basis of the fingerprint |
| `fingerprint` | text | sha256 of (tool, canonical inputs) or canonical claim; used for reuse |
| `tool_name`, `tool_inputs` | text, JSON | what produced it (Code tab, "Download script") |
| `result` | JSON | typed: `{kind: expr|number|array|table|bool|plotspec, value, units?}` |
| `status` | enum | `proposed`, `verified`, `failed`, `invalidated`, `hypothesis` (see 4.1 for `pinned`) |
| `pinned` | bool | user-protected; see 4.1 |
| `needs_recheck` | bool | set on unverified nodes loaded from earlier sessions (shown as hints) |
| `confidence` | real or null | from tool metadata (error estimates, solver warnings, run disagreement), never from the model's self-report |
| `risk` | JSON | which rules fired and which checks were offered |
| `ladder_stage` | enum | `none`, `retried`, `backtracked`, `escalated` |
| `role` | text | worker role that produced it |
| `superseded_by` | text or null | next version's id |
| `created_at`, `updated_at` | timestamps | |

Dependencies live in the `edges` table, and evidence in the `evidence` table (both below), so they are queryable instead of buried in JSON.

### 3.3 Status rules (enforced in `graph/engine.py`, not by the model)

- Only the verifier can set `verified`, and only with at least one passing evidence row from an independent method.
- `chem` nodes are always `hypothesis`; any attempt to set `verified` raises.
- Setting `failed` or `invalidated` triggers `cascade.py`: every node that transitively `depends_on` it becomes `invalidated`, in every session (the DB is global). Each change is logged in `status_events`.
- `invalidated` is terminal for a version; recovery creates a new version in the same lineage.

### 3.4 SQLite tables

```sql
sessions(id PK, title, created_at, updated_at)

nodes(id PK, lineage_id, version, session_id, layer, type, domain, title,
      content, content_canonical, fingerprint, tool_name, tool_inputs JSON,
      result JSON, status, pinned, needs_recheck, confidence, risk JSON,
      ladder_stage, role, superseded_by, created_at, updated_at,
      UNIQUE(lineage_id, version))
  INDEX(fingerprint, status), INDEX(session_id), INDEX(lineage_id)

edges(src, dst, kind, meta JSON, PRIMARY KEY(src, dst, kind))
  -- kind: depends_on | checks | conflicts_with | relates | renders
  -- "src depends_on dst": src was computed from dst

evidence(id PK, node_id, method, tool_name, inputs JSON, outcome, detail JSON, created_at)
  -- method: symbolic_vs_numeric | alt_algorithm | known_value | units | sanity
  -- outcome: pass | fail | inconclusive

tool_runs(id PK, session_id, node_id, tool_name, inputs JSON, output JSON,
          stderr, duration_ms, error, created_at)          -- Log tab

messages(id PK, session_id, node_id NULL, author, text,
         pin_number NULL, pin_x NULL, pin_y NULL, sent_to_controller, created_at)
                                                            -- chat, node Chat tab, pins

status_events(id PK, node_id, from_status, to_status, reason, caused_by_node, created_at)
```

### 3.5 Controller actions (the only things the model may emit)

```json
{"action":"call_tool","thought":"<=160 chars","tool":"sympy.integrate",
 "args":{"expr":"x**2*sin(x)","var":"x"},"depends_on":["n3"],"title":"integrate by parts"}
{"action":"run_check","node":"n5","check":"symbolic_vs_numeric"}
{"action":"finish","answer_template":"The antiderivative is {{n5}}.","answer_nodes":["n5"]}
{"action":"ask_user","question":"..."}
```

The offered checks ride in the next observation, so choosing a check costs no extra model call. When only one check is offered, code runs it without asking the model.

### 3.6 "The model never produces a numeric result"

Two code guards, no trust in prompting:
1. The final answer is a template; `{{n5}}` is replaced by node `n5`'s `result`. A number literal outside a placeholder is rejected unless the question itself contains it (so `y(2) = {{n5}}` passes and `the answer is 42` does not): one retry, then an error node. A recipe's answer sentence is written by code, so there is no model prose in it at all.
2. `call_tool` args may contain numbers only if they come from the problem statement or an existing node; otherwise the action is rejected.

### 3.7 Reuse lookup (acceptance test 3)

`controller/lookup.py` runs first, with no model call: normalized problem text and its fingerprint are matched against `verified` nodes. On a hit, the verified node is linked into the session and answered from. If the problem needs translation to a formal form, the model may be called once to formalize it, but never to do the math; the lookup then matches on the fingerprint. Unverified matches come back as `hint` nodes with `needs_recheck=1`.

Phase 4 widens this one step, and only for `chem`: a chemistry node is never `verified`, so for those the test is that every check that ran passed and none failed. Such a node is linked in and reused exactly as a verified one would be, and it stays a `hypothesis` there, so the answer built on it is not verified either (`controller/lookup.py: reusable_by_fingerprint`).

### 3.7a Recipe library and router (2026-10-07)

Formalization used to be open ended: the model wrote a free-form `goal` tool call and described the question's values in `givens`, which only accepted quantity objects (`{"value", "unit", "kind"}`). Live checks on qwen2.5, qwen3 8B and 14B, mistral-nemo and phi4 all failed at the same step, and the larger models failed harder. Two causes, both in the schema rather than the model:

- A "given" that is not a number had nowhere to go. An integrand, the right-hand side of a differential equation, the curves of an area problem are expressions, and the validator answered `a quantity is {"value": number, "unit": "m/s"}` or `unknown kind`. No reply could pass.
- An open-ended goal means the model picks the formula, so the answer was only as good as the model's memory of it.

`controller/recipes.py` replaces it. A recipe is a fixed list of named slots (`number`, `unit`, `expr`, `equation`, `bound`, `var`, `choice`, `numbers`, `derivative`), the tool calls that answer the question, the problem type whose assumption checklist it carries, and two worked examples. The router is one model call whose `recipe` field is an enum: the nine recipes, `dataset_question` and `molecule_question` (which hand the question to the Phase 3 and Phase 4 paths unchanged), and `none`. Code coerces each slot ("20" and 20 alike, "1,000", "3/4", `-2*y` or `-2*y(x)`, "20 m/s" split across a number and its unit slot), re-prompts only on a real type error, and asks the *user* when a required slot is a value the question never contained. `scripts/live_check.py` scores a planning-call budget per problem, so "one model call for an easy question" is measured, not asserted.

Four closed-form solvers were added for the recipes, each with a checker that reaches the same number another way (`tools/formula_tools.py`): `phys.projectile_range` (checked by integrating the trajectory with RK4 and bisecting the ground crossing), `phys.photon_energy` (checked against pint's own constant table rather than SciPy's CODATA), `phys.rc_discharge` (checked by solving `dV/dt = -V/RC` numerically) and `calc.area_between` (crossings, then the symbolic integral of each `|f-g|` piece, checked by quadrature on a 4001-point grid, and refusing an area that is negative or zero). `units.check_target` is a `required`, `must_pass` plan on every solver that takes a `to_unit`: it fails a wrong dimension (a photon energy reported as `1 / m J`) and a right-dimension-wrong-unit result (96560.64 m/h for a question asking m/s) separately, because both read as the answer. `ode.check_residual` is now `must_pass`, so a solution that does not satisfy its own initial condition cannot be verified.

Display: values show to four significant figures, the unit is written exactly once (by the result, never by the sentence), and `units.convert` reports the unit the question asked for rather than pint's spelling of it.

Not changed: numbers still only come from tools, answers are still templates filled from node results, every tool result still gets an independent check, and a chemistry node is still always a hypothesis. A recipe's slots are the problem's sources, so a slot holding a number the question never had still flags every result built on it.

### 3.7b One schema for the request and the reply (2026-10-07)

The first live run of the recipe branch failed every problem of all three suites at the same step: `formalization failed: formalize needs field(s): statement`, twice per problem, 0/6, 0/6 and 0/8. The recipes were not the problem and neither was the model. The request sent `ACTION_SCHEMA` to Ollama as the output format with `"required": ["action"]`, so the grammar the model generated under permitted `{"action","recipe","slots"}` and the model stopped there, exactly as a constrained model does. The validator, a separate table in the same file, demanded `statement` as well. The 629 mock tests passed throughout, because a mock emits whatever the test wrote.

Three changes, so neither half can drift from the other again:

- `llm/actions.py` defines each field once (`PROPERTIES`), which action may carry it (`FIELDS`) and which must (`REQUIRED`). `action_schema(allowed)` builds from those: one allowed action is a plain object whose `required` list forces its fields, several are an `anyOf` of one such object per action. `_ask` builds it once, passes it to `chat` as the output format and passes the same object to `parse_action`, which validates against the variant inside it. The only requirement check made on a reply is the schema's own `required` list, so a reply the request permitted cannot be rejected.
- `statement` is no longer required of the model. `Recipe.statement(values)` writes it from the recipe and its filled slots, which is more precise than a restatement, and the model's own wording is still preferred when it sends one. What the router must now produce is `recipe` and `slots`, the two things only it can decide.
- The scripted model in `tests/fakes/fake_llm.py` validates every scripted reply against the schema the controller handed it. A mock may only say what a real model could say, so a drift like this one now fails the whole suite at once rather than passing it. `tests/fixtures/live/` holds recorded replies in the format `--trace` writes, one per route, and `tests/acceptance/test_live_contract.py` checks each against the schema, the validator, the planner and the controller, plus the *minimal* object the schema permits for every action -- which is the shape that failed live.

`scripts/live_check.py --trace` writes every exchange to `results/trace-<suite>-<n>.jsonl`: the system prompt, the user prompt, the JSON schema sent as the output format, and the raw reply. A formalization failure needs all four in one place; this bug was invisible from the table alone.

Declines moved too. A question plainly asking for an operation nothing performs (a synthesis route, a dose) is now declined in `_run` before any model call, by `chem_decline.decline_for_question`. The capability rule is unchanged: patterns only name the operation a question asks for, `served()` still decides, and a question asking for anything a tool does is never pre-declined -- so "the logP of the product of this synthesis" stays a logP question. It matters because while formalization was broken those two requests came back as errors; a refusal must not depend on the model replying at all.

### 3.7c Slot provenance, and the router guard (2026-10-07)

With formalization fixed, the second live run answered maths 6/6 and physics 6/6 in one model call each, and chemistry 2/8. The two passes were the two declines. Every other chemistry question came back with a physics number: caffeine's molecular weight, ibuprofen's logP and paracetamol's drug-likeness were all "answered" 3.9728917e-19 (the photon-energy example's own value), the similarity question came back 0.06, 0.06, 0.03 (the circuit example's currents), and the repeat dutifully reused the stored wrong answer. The router had sent molecule questions to physics recipes, and the model, with no numbers of its own to copy, filled the slots from the worked examples in the prompt.

Nothing in the design said a slot had to come from the question. The prompt now says so, but a prompt is not a guarantee, so the rule is code:

- **Provenance is a gate, not a flag.** `PROVENANCE` in `controller/recipes.py` maps a slot's kind to the rule that sources it: `numbers` (every number in the value must appear in the question), `text` (a structure or library member must appear in it verbatim) or `none` (a unit or a variable name carries no data). `Recipe.unsourced(plan, question)` returns the slots that fail, `Plan` carries them, and `_formalize` raises `UnsourcedSlots`: one re-prompt naming the offending slots, then the question is answered out of scope. `_run_recipe` refuses any plan with `unsourced` as its last gate before the tools, so no path reaches a tool with a value the question never had. `controller/provenance.py` reads number words as well as digits, so "x squared" sources the 2 in `x**2` and "a dozen" sources 12.
- **The router guard is code too.** `domains/chem/routing.looks_chemical(question)` fires on any of three independent signals: a chemistry term, a molecule name, or SMILES punctuation. A question it recognizes may only go to one of the five `chem_` recipes, to `molecule_question` or to `none`; a physics or maths recipe raises `WrongRoute` and the question is answered out of scope. "Compound interest" and "the structure of the beam" are not chemical, and the test says so.
- **An unsourced result is never stored and never reused.** `repository.purge_unsourced()` runs once per store, deletes the problem roots of runs whose slots failed provenance and everything derived from them, and records that it has done so in `meta`; the UI says how many were removed. `controller/lookup.py` filters reuse, fingerprint reuse and hints through `unsourced_root`, so a wrong answer already in a store cannot come back. `live_check.py` runs on a fresh temporary store by default (`--fresh-store`, with `--no-fresh-store` to use the configured one), so a suite neither inherits nor leaves behind a stored answer.
- **The chemistry tools are recipes.** `chem_identity`, `chem_descriptors`, `chem_logp`, `chem_druglike` and `chem_similarity` have slots, steps and checks like every other recipe, so a molecule question costs the one routing call and its structure is a slot the provenance rule can police. `chem_druglike` runs descriptors, then logP, then the filters, from that single call. Every chemistry node is still a hypothesis, the decline path is unchanged, and the `molecule_question` goal path still exists for substructure, clustering, ranking and literature.
- **Rounding is half up.** `graph/model._round_half_up` uses `Decimal` with `ROUND_HALF_UP`, because Python's `round` is half-to-even and turned 298.15 K into 298.1. A temperature keeps two decimals rather than four significant figures, so 25 degC converts to 298.15 K.

Each of the five live failures is a permanent test. `tests/fixtures/live/` carries the recorded replies, including the three the model actually sent, and `test_a_worked_example_value_never_reaches_an_answer` replays *every* worked example of *every* recipe against a question that contains none of its values: the plan must be refused, no tool may run, and no value of the example may appear in what the user is told.

**A structure is compared as a molecule, not as a string.** The next live run scored chemistry 6/8, and both failures were the same caffeine question: the router chose `chem_descriptors` correctly, and the model then wrote caffeine's Kekule form, `CN1C=NC2=C1C(=O)N(C)C(=O)N2C`, instead of copying the aromatic SMILES the question gave. The same molecule, so the question's own structure, refused by a text match. The same text match was also too generous in the other direction: `C` is a substring of almost any question carrying a SMILES, so methane passed provenance as the molecule asked about.

The `structure` rule therefore asks whether the question contains a structure that *is* this molecule. `domains/chem/standardize.keys_in_text` scans the question for runs of SMILES-legal characters -- which splits prose and a JSON library alike -- reads each candidate with the same `read_twice` gate every structure goes through, and returns the InChIKeys found; `in_text` compares the slot's own key against that set. A run holding a letter the organic subset cannot write outside brackets is a word, not a structure, and never reaches the toolkit, so a question's prose costs nothing and prints nothing. A molecule the question does not contain still has nothing to match, and a name is not a structure, so a question naming only "caffeine" sources no SMILES the model remembers.

A structure copied out of prose also arrives with the prose: the question writes "...caffeine, Cn1cnc2c1c(=O)n(C)c(=O)n2C?" and "...aspirin (CC(=O)Oc1ccccc1C(=O)O)?", so the slot's coercion trims the sentence's punctuation and any brackets or quotes around the structure. None of that can begin or end a SMILES, while the charset guard has to stay wide enough for InChI, which does use `?` and `,` -- so text beginning `InChI=` is left exactly as written.

**A lone structure is read out of the question, not copied by the model.** The next run, with all of the above in place, still refused the same caffeine question and nothing else: qwen3:8b answered identity, logP, drug-likeness and the similarity ranking in one call each, three runs in a row, and got caffeine wrong every time. A structure was the one value still being transcribed by the model -- twenty-six characters of brackets, digits and case -- in a system whose whole point is that code builds the tool calls from the question's own values.

So it does that here too. `Recipe.read_structures` asks `domains/chem/standardize.structures_in_text` what structures the question itself contains; when there is exactly one, that is the question's answer to "which molecule", and it fills every `smiles` slot regardless of what the model wrote. The slot leaves `supplied`, because a value code read out of the question is a source rather than a claim to trace, and `Plan.from_question` records that it happened, so the root node the user confirms says "(structure read from the question)" and the live check's row shows it. A question carrying several structures -- a similarity query and its library -- still needs the model to say which one is the subject, so that choice is traced exactly as before.

**A question that draws no structure is asked about, not refused.** "What is the logP of caffeine?" names a molecule and writes none, so the only structure available is one the model remembers, which is not a source. Out of scope was the honest answer and a poor one: the system knows exactly which recipe fits and exactly which value is missing. `Recipe.read_structures` now raises `MissingSlots` when the question contains no structure at all, `loop._needs_slots` records a `Missing value` hint naming the recipe and the slots, and the task comes back `needs_user` asking "What is the structure of the molecule, as SMILES, InChI or a molblock?". A similarity question, which needs two, asks for both in one sentence. No model call is spent on the re-prompt, because re-prompting cannot produce a value the question never had.

That is why a slot has two descriptions. `Slot.about` is written for the model and says where to find the value ("copied from the question exactly as it is written"), which is useless as a question to a person; `Slot.ask` is the user-facing phrase, and `asked()` falls back to `about` for the slots whose wording suits both. The structure the user then supplies is part of the question on the next turn, so it is read by code like any other -- the ask changes where the value comes from, not the rule.

**All three formats are read out of the question, not just SMILES.** Asking for "SMILES, InChI or a molblock" exposed a gap in the reader behind the ask: the tools read all three, and the `format` slot names all three, but the question scanner only found SMILES. A question written with an InChI was therefore unsourceable -- refused before, and now asked about in a loop, since the answer would not be readable either. Neither of the other two formats survives a scan for SMILES-legal runs: an InChI is split on its commas, and a molblock's coordinate lines each end in a bare element symbol, so `C` in the block would be read as methane. So `_candidate_groups` finds molblocks first (three header lines, the counts line, through `M  END`), then InChIs (`InChI=1S/...`), blanks each out of the text it scans next, and only then collects the SMILES runs. `fmt_of` picks the reader for a candidate, and `as_smiles` gives back the SMILES of a structure however the question wrote it, so a structure code reads is handed to the tools in one format and the `format` slot is corrected to say `smiles` -- the model may have set it to match the question's wording, which is no longer what the slot holds.

Two smaller bugs in the same path, both pre-existing: `standardize.parse` stripped the text it was given, and a molblock's first line is its title and is usually blank, so stripping shifted every line up by one and the counts line was read as coordinates; and `MAX_STRUCTURE`, the slot's length cap, was written for a SMILES at 1000 characters, which refuses the molblock of any real molecule. A molblock is a line per atom and a line per bond, so it has its own cap (`MAX_MOLBLOCK`) sized to the 400-atom limit the toolkit itself enforces.

**Asking the same molecule question twice costs nothing.** With everything above in place the suite answered maths 6/6, physics 6/6 and chemistry 7/8, the last row being the repeat: it was answered correctly, but by routing and recomputing it. The lookup that answers a question from the store wanted a *verified* final, and a chemistry answer is never verified -- that is the rule of the phase -- so no molecule question could ever be answered from the store, however many times it was asked.

`lookup.answer_for_question` applies chemistry's own test instead, the one its tool-level reuse was already given in Phase 4: every tool result the answer rests on passed its checks and none failed. What that buys is reuse, not belief, so the answer comes back exactly as it was stored -- `verified=False`, flagged `hypothesis`, and its detail says "answered from a stored hypothesis whose checks passed; it is still a hypothesis" rather than claiming a verified store hit. A check that fails afterwards withdraws the reuse and the next asker recomputes. The walk to the results skips the problem node and the assumptions, which are not results and have nothing to check.

**A refusal says which refusal it was.** Two chemistry runs in a row came back 6/8 with a row reading only "I can only answer questions that fit one of the recipes I have", which is the sentence *every* out-of-scope question gets, so the table could not tell `WrongRoute` from an unsourced slot from `recipe: none`, and diagnosing it meant asking for a trace file off the user's machine. A failing `live_check` row now carries what the graph recorded -- the route that could not answer, or the slot and the value the model put in it -- and the saved report quotes every failing problem's raw replies under the table. The sentence the user reads is unchanged; this is the suite reading the graph, not the graph telling the user more.

### 3.8 Failure ladder (in `verify/ladder.py`)

1. Check fails -> retry once with a different method or role (new version of the same lineage).
2. Still failing -> mark the node and all dependents `invalidated`, re-derive from the deepest `verified` ancestor.
3. Still conflicting -> stop, add both results with a `conflicts_with` edge, show them side by side, ask the user.

## 4. Decisions I need from you

1. **`pinned` as a flag, not a status.** A pinned node still needs to say whether it is verified or failed, so I propose `pinned` as a separate boolean. *Recommended: flag.*
2. **Final answer as a template with node references** (3.6). This is the strongest way to guarantee the model never types a number. *Recommended: yes.*
3. **Where the code lives.** No repository is attached to this project. Options: a new GitHub repo you create and attach (recommended, so you get PRs per phase), or code delivered into the project files folder.

Everything else follows your spec as written. On approval I start Phase 1 in this order: graph engine + store, tools + sandbox, risk rules + verifier + ladder, controller with the fake LLM, acceptance tests, then the minimal UI.

## 5. Approved changes (2026-10-03)

1. The node flag is `locked` (pins stay as comments). Locked nodes are never auto-deleted or auto-demoted, but still show failed/invalidated, with a warning.
2. Number provenance: the formalized problem is the root node, shown to the user for confirmation. Tool args may use numbers from any ancestor node plus small structural integers (|n| <= 10); other numbers flag the node for the verifier instead of being rejected. Final-answer templates keep the hard reject on digits.
3. Safe parsing: a token whitelist plus a restricted namespace (no builtins) in front of `parse_expr`, an unevaluated pre-parse that rejects size bombs, and the sandbox process as the second layer.
4. A SQLite CHECK constraint and an UPDATE trigger block `chem` + `verified`. A node's domain comes only from the tool registry (or is derived from its dependencies), never from the caller.
5. `num_ctx` is set explicitly on every Ollama request (default 8192) and the graph digest is sized to fit it; tested with an 800-node graph at 8192 and 4096.
6. A cascade that reaches nodes used by other sessions returns a report of the affected sessions and answers, shows it in the UI, and stores a notice that appears when those sessions are opened.

Decisions: `locked` flag; template-only final answers; code in the private `science-ai` repo with one branch per phase.
