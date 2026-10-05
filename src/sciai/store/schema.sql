-- Science AI knowledge store, schema version 4 (v2 adds the 'assumption' node type,
-- v3 adds the datasets table, v4 adds the molecules table). Older stores are migrated on
-- open by sciai.store.migrate.
-- The database is global across sessions: cascades cross session boundaries.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id                TEXT PRIMARY KEY,
    lineage_id        TEXT NOT NULL,
    version           INTEGER NOT NULL CHECK (version >= 1),
    session_id        TEXT NOT NULL REFERENCES sessions(id),
    layer             TEXT NOT NULL CHECK (layer IN ('reasoning','domain','plot')),
    type              TEXT NOT NULL CHECK (type IN ('problem','claim','step','tool_result','check',
                                                    'final','error','hint','entity','plot',
                                                    'assumption')),
    domain            TEXT NOT NULL CHECK (domain IN ('general','math','physics','data','chem')),
    title             TEXT NOT NULL,
    content           TEXT NOT NULL DEFAULT '',
    content_canonical TEXT NOT NULL DEFAULT '',
    fingerprint       TEXT NOT NULL DEFAULT '',
    tool_name         TEXT,
    tool_inputs       TEXT,           -- JSON
    result            TEXT,           -- JSON
    status            TEXT NOT NULL CHECK (status IN ('proposed','verified','failed','invalidated','hypothesis')),
    locked            INTEGER NOT NULL DEFAULT 0 CHECK (locked IN (0,1)),
    needs_recheck     INTEGER NOT NULL DEFAULT 0 CHECK (needs_recheck IN (0,1)),
    confidence        REAL,
    risk              TEXT NOT NULL DEFAULT '{}',   -- JSON
    flags             TEXT NOT NULL DEFAULT '[]',   -- JSON
    ladder_stage      TEXT NOT NULL DEFAULT 'none' CHECK (ladder_stage IN ('none','retried','backtracked','escalated')),
    role              TEXT,
    superseded_by     TEXT,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    UNIQUE (lineage_id, version),
    -- Drug-discovery safety rule: a chemistry node can never be verified.
    CHECK (NOT (domain = 'chem' AND status = 'verified'))
);

-- Belt and braces for the chem rule: also refuse it on UPDATE paths, with a clear message.
CREATE TRIGGER IF NOT EXISTS chem_never_verified
BEFORE UPDATE OF status, domain ON nodes
WHEN NEW.domain = 'chem' AND NEW.status = 'verified'
BEGIN
    SELECT RAISE(ABORT, 'chem nodes can never be verified');
END;

CREATE INDEX IF NOT EXISTS idx_nodes_fingerprint ON nodes(fingerprint, status);
CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_nodes_lineage ON nodes(lineage_id, version);

-- Which nodes appear in which session, with the short handle (n1, n2...) the
-- controller uses. Reused nodes from other sessions are linked, not copied.
CREATE TABLE IF NOT EXISTS session_links (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    node_id    TEXT NOT NULL REFERENCES nodes(id),
    handle     TEXT NOT NULL,
    PRIMARY KEY (session_id, node_id),
    UNIQUE (session_id, handle)
);
CREATE INDEX IF NOT EXISTS idx_links_node ON session_links(node_id);

CREATE TABLE IF NOT EXISTS edges (
    src  TEXT NOT NULL REFERENCES nodes(id),
    dst  TEXT NOT NULL REFERENCES nodes(id),
    kind TEXT NOT NULL CHECK (kind IN ('depends_on','checks','conflicts_with','relates','renders')),
    meta TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (src, dst, kind),
    CHECK (src <> dst)
);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst, kind);

CREATE TABLE IF NOT EXISTS evidence (
    id            TEXT PRIMARY KEY,
    node_id       TEXT NOT NULL REFERENCES nodes(id),
    check_node_id TEXT REFERENCES nodes(id),
    method        TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    inputs        TEXT NOT NULL,
    outcome       TEXT NOT NULL CHECK (outcome IN ('pass','fail','inconclusive')),
    detail        TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_node ON evidence(node_id);

CREATE TABLE IF NOT EXISTS tool_runs (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    node_id     TEXT,
    tool_name   TEXT NOT NULL,
    inputs      TEXT NOT NULL,
    output      TEXT,
    error       TEXT,
    duration_ms REAL NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_node ON tool_runs(node_id);

CREATE TABLE IF NOT EXISTS messages (
    id                 TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    node_id            TEXT,
    author             TEXT NOT NULL,       -- user | controller | system
    text               TEXT NOT NULL,
    pin_number         INTEGER,
    pin_x              REAL,
    pin_y              REAL,
    sent_to_controller INTEGER NOT NULL DEFAULT 0,
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, node_id);

CREATE TABLE IF NOT EXISTS status_events (
    id             TEXT PRIMARY KEY,
    node_id        TEXT NOT NULL,
    from_status    TEXT,
    to_status      TEXT NOT NULL,
    reason         TEXT NOT NULL,
    caused_by_node TEXT,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_node ON status_events(node_id);

-- Cross-session cascade notices, shown when the affected session is opened.
CREATE TABLE IF NOT EXISTS notices (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    origin_node  TEXT NOT NULL,
    text         TEXT NOT NULL,
    detail       TEXT NOT NULL,
    seen         INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notices_session ON notices(session_id, seen);

-- Imported data files (v3). A dataset is immutable and content-addressed: ``id`` is a hash
-- of the file's SHA-256, its format and the parse options, so the same bytes read the same
-- way are always the same dataset, and a changed file is a new one. The file is copied to
-- ``stored_path`` (named by its SHA-256) and never read from its original path again.
-- Several rows may share a name; the most recently imported one is the current dataset.
CREATE TABLE IF NOT EXISTS datasets (
    id            TEXT PRIMARY KEY,
    sha256        TEXT NOT NULL,
    name          TEXT NOT NULL,
    format        TEXT NOT NULL CHECK (format IN ('csv','tsv','xlsx')),
    options       TEXT NOT NULL,      -- JSON: delimiter, decimal mark, encoding, sheet, NA tokens
    stored_path   TEXT NOT NULL,
    original_path TEXT,
    rows          INTEGER NOT NULL CHECK (rows >= 0),
    columns       INTEGER NOT NULL CHECK (columns >= 0),
    schema        TEXT NOT NULL,      -- JSON: [{name, type, unit, missing, levels?}]
    imported_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name, imported_at);

-- Phase 4. One row per standardized structure the store has seen, keyed by InChIKey so a
-- stereoisomer is its own row. skeleton is the key's first block, shared by the salt forms and
-- stereoisomers of one substance, which is what reuse and the restricted screen key on.
-- There is deliberately no status column here: a molecule record is bookkeeping, and every
-- claim about a molecule lives in nodes, where the chem rule keeps it a hypothesis.
CREATE TABLE IF NOT EXISTS molecules (
    inchikey        TEXT PRIMARY KEY,
    skeleton        TEXT NOT NULL,
    canonical_smiles TEXT NOT NULL,
    inchi           TEXT NOT NULL,
    formula         TEXT NOT NULL,
    atoms           INTEGER NOT NULL CHECK (atoms > 0),
    bonds           INTEGER NOT NULL CHECK (bonds >= 0),
    charge          INTEGER NOT NULL,
    name            TEXT,                -- what the user called it, if they named it
    input_form      TEXT NOT NULL,       -- the structure as it was supplied
    standardized    TEXT NOT NULL,       -- JSON: the steps standardization took, in order
    screen          TEXT NOT NULL,       -- JSON: the restricted screen's verdict on the way in
    source_file     TEXT,                -- the imported file it came from, if any
    source_sha256   TEXT,
    members         INTEGER,             -- set for a library record, null for one structure
    imported_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_molecules_skeleton ON molecules(skeleton);
CREATE INDEX IF NOT EXISTS idx_molecules_name ON molecules(name, imported_at);
