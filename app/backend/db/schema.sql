PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- documents
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT    PRIMARY KEY,  -- UUID assigned at upload time
    filename    TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    page_count  INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL DEFAULT 'ingesting',  -- ingesting | ready | failed
    error       TEXT,                                  -- populated on failure
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------------------------------------------------------------------------
-- chats
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chats (
    id         TEXT PRIMARY KEY,  -- UUID
    title      TEXT NOT NULL DEFAULT 'New chat',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------------------------------------------------------------------------
-- messages
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,  -- UUID
    chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,     -- user | assistant
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);

-- ---------------------------------------------------------------------------
-- message_sources
-- Cited chunks attached to an assistant message.
-- chunk_index is the 1-based [n] marker position used in the answer text.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS message_sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  TEXT    NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,  -- 1-based; matches [n] in answer text
    doc_id      TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    page_start  INTEGER,
    page_end    INTEGER,
    snippet     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_message_sources_message_id ON message_sources(message_id);

-- ---------------------------------------------------------------------------
-- message_traces
-- Thinking-trace steps attached to an assistant message.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS message_traces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT    NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,  -- display order (0-based)
    step       TEXT    NOT NULL,
    status     TEXT    NOT NULL,  -- done | error
    info       TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_message_traces_message_id ON message_traces(message_id);

-- ---------------------------------------------------------------------------
-- collections
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS collections (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS document_collections (
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    PRIMARY KEY (doc_id, collection_id)
);
