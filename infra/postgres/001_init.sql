CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS task_definitions (
    task_id TEXT NOT NULL,
    version TEXT NOT NULL,
    definition JSONB NOT NULL,
    definition_sha256 CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (task_id, version)
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    code_commit TEXT,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    episode_id TEXT NOT NULL,
    task_id TEXT,
    task_version TEXT,
    status TEXT NOT NULL CHECK (status IN (
        'queued', 'running', 'succeeded', 'failed', 'cancelled', 'timed_out', 'truncated'
    )),
    trace_id CHAR(32),
    raw_prefix TEXT,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (experiment_id, episode_id),
    FOREIGN KEY (task_id, task_version) REFERENCES task_definitions(task_id, version)
);

CREATE INDEX IF NOT EXISTS runs_experiment_status_idx ON runs (experiment_id, status);
CREATE INDEX IF NOT EXISTS runs_task_idx ON runs (task_id, task_version);

CREATE TABLE IF NOT EXISTS pipeline_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kind TEXT NOT NULL CHECK (kind IN ('normalize', 'episodes', 'evaluate', 'dataset', 'compact')),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    source_manifest TEXT,
    output_manifest TEXT,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    error JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS pipeline_jobs_status_idx ON pipeline_jobs (status, created_at);

CREATE TABLE IF NOT EXISTS dataset_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('eval', 'sft', 'preference', 'rl')),
    manifest_ref TEXT NOT NULL UNIQUE,
    manifest_sha256 CHAR(64) NOT NULL,
    source_snapshot_id TEXT NOT NULL,
    row_count BIGINT NOT NULL CHECK (row_count >= 0),
    frozen BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS dataset_snapshots_name_idx ON dataset_snapshots (name, created_at DESC);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id TEXT NOT NULL,
    event_id TEXT,
    annotator_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    label JSONB NOT NULL,
    confidence DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS annotations_episode_idx ON annotations (episode_id, created_at);
