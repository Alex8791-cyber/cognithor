-- Jarvis · PostgreSQL Schema
-- Verwendet pgvector fuer native Vector-Suche und tsvector fuer BM25.

CREATE EXTENSION IF NOT EXISTS vector;

-- Chunks-Tabelle (entspricht SQLite chunks)
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source_path TEXT NOT NULL,
    line_start INTEGER DEFAULT 0,
    line_end INTEGER DEFAULT 0,
    content_hash TEXT NOT NULL,
    memory_tier TEXT NOT NULL DEFAULT 'working',
    entities_json TEXT DEFAULT '[]',
    timestamp TIMESTAMPTZ,
    token_count INTEGER DEFAULT 0,

    -- tsvector fuer BM25-aehnliche Suche
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('german', text)) STORED
);

-- GIN-Index fuer Volltext-Suche (ersetzt FTS5)
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks (source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_tier ON chunks (memory_tier);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks (content_hash);

-- Embeddings-Tabelle mit pgvector
CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT PRIMARY KEY,
    embedding vector(768) NOT NULL,
    model TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW-Index fuer Vector-Suche (Cosine Distance)
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 32, ef_construction = 200);

-- Entities-Tabelle
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    attributes_json TEXT DEFAULT '{}',
    source_file TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    confidence REAL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities (name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (type);

-- Relations-Tabelle
CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    source_entity TEXT NOT NULL REFERENCES entities(id),
    relation_type TEXT NOT NULL,
    target_entity TEXT NOT NULL REFERENCES entities(id),
    attributes_json TEXT DEFAULT '{}',
    source_file TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    confidence REAL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON relations (source_entity);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations (target_entity);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations (relation_type);

-- Sessions-Tabelle
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    agent_name TEXT DEFAULT 'jarvis',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    message_count INTEGER DEFAULT 0,
    active BOOLEAN DEFAULT TRUE,
    data_json TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id, channel);

-- Chat-History
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    channel TEXT DEFAULT '',
    name TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history (session_id, timestamp);

-- Additional performance indexes
CREATE INDEX IF NOT EXISTS idx_chunks_timestamp ON chunks(timestamp);
CREATE INDEX IF NOT EXISTS idx_chunks_tier_timestamp ON chunks(memory_tier, timestamp);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_ts ON chat_history(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp ON chat_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity);
CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source_file);
