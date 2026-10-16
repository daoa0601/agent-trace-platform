# Architecture

## Separation of concerns

The collector performs bounded work only: authenticate, validate, redact, offload oversized content,
compress, and append. It does not join history, execute evaluators, update rewards, or build episodes.
This makes a retry safe and keeps collection latency independent of task complexity.

Offloading applies to free-form attributes as well as typed content. OTLP instrumentation often puts
prompts and responses in span attributes (`input.value`, `llm.input_messages.*`). When a value, or
the event's attributes together, exceed `inline_payload_bytes`, the largest values move to the blob
store and the event lists them under `offloaded_attributes` by reference.

Raw object storage is the audit source of truth. A canonical snapshot is a deterministic projection of
a set of raw object digests. An episode snapshot is a deterministic projection of a canonical manifest.
Evaluation and consumer datasets repeat that pattern. A trainer or benchmark records one frozen
dataset manifest rather than a mutable query.

## Storage layers

| Layer | Format | Mutability | Purpose |
|---|---|---:|---|
| raw | JSONL + Zstandard | append-only | audit, reparsing, instrumentation recovery |
| blobs | content-addressed JSON + Zstandard | immutable/deduplicated | prompts, tool output, token arrays, state |
| canonical | Parquet snapshot | immutable | typed filtering, joins, data-quality checks |
| episodes | Parquet with lossless episode JSON | immutable | ordered trajectories and quarantine |
| evaluations | Parquet | immutable | versioned labels and decomposed rewards |
| datasets | Parquet + manifest | frozen | eval, SFT, preference, or RL consumer contract |
| audit bundles | content-addressed JSON; derived Markdown | JSON immutable | benchmark claims, evidence, independent re-tests, publication gate |
| operational | PostgreSQL | mutable | tasks, run state, jobs, annotations, registrations |

Canonical rows deliberately contain both stable scalar columns and `event_json`. Scalar columns make
DuckDB/Trino/Spark scans cheap; the lossless JSON preserves forward-compatible reconstruction until a
field earns its own analytical column.

## Scale path

The local deployment writes whole immutable snapshots. When volumes require incremental tables,
register the same Parquet schemas in Apache Iceberg and use snapshot IDs from the Iceberg catalog.
Kafka/Flink may validate live traffic, but globally consistent episode, reward, filtering, and dataset
jobs remain batch operations.

Auditable benchmark sessions are a side branch from the same immutable trace source. They bind a
versioned task and protocol to hardware observation, a final artifact, raw metric evidence, and the
full trace digest. An independent audit and its empirical re-tests are processed offline; no reviewer
or publication-gate work runs on the collector request path. `benchmark-audit-core` owns the portable
manifest and decision semantics; AgentTrace owns the trace and object-store integrations. Markdown is
a deterministic view of the immutable JSON bundle, not a separately persisted source of truth.

## Failure semantics

- Native client retries are idempotent because identical compressed batches resolve to the same key.
- The SDK can spool failed batches locally and replay them later.
- OTLP Collector retry queues absorb temporary collector failures.
- An invalid record cannot poison an entire canonical snapshot; it is written to quarantine.
- Conflicting records with the same `event_id` are both excluded.
- A broken content reference is an error in strict normalization.
- An invalid episode is retained in `quarantined_episodes`, never silently dropped.
