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
1. The final answer is a template; `{{n5}}` is replaced by node `n5`'s `result`. Any number literal in the model's prose that is not in a referenced node's inputs or result rejects the action (one retry, then an error node).
2. `call_tool` args may contain numbers only if they come from the problem statement or an existing node; otherwise the action is rejected.

### 3.7 Reuse lookup (acceptance test 3)

`controller/lookup.py` runs first, with no model call: normalized problem text and its fingerprint are matched against `verified` nodes. On a hit, the verified node is linked into the session and answered from. If the problem needs translation to a formal form, the model may be called once to formalize it, but never to do the math; the lookup then matches on the fingerprint. Unverified matches come back as `hint` nodes with `needs_recheck=1`.

Phase 4 widens this one step, and only for `chem`: a chemistry node is never `verified`, so for those the test is that every check that ran passed and none failed. Such a node is linked in and reused exactly as a verified one would be, and it stays a `hypothesis` there, so the answer built on it is not verified either (`controller/lookup.py: reusable_by_fingerprint`).

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
