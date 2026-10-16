# AgentTrace project context

AgentTrace is a general agent-data platform, not a coding benchmark. Keep task semantics,
environment setup, trace collection, evaluation, and dataset export independent. Benchmark runners,
model-provider harnesses, and audit gating do not belong here; downstream code uses only the public
modules listed in the README. It needs the sibling `benchmark-audit-core` checkout (see the catalog
README for the layout).

The immutable raw object store is the source of truth. Never rewrite raw batches in place.
Corrections produce a new canonical snapshot, and every exported dataset records its source
snapshot and content digest.

Large payloads belong in the content-addressed blob store. Canonical Parquet rows contain typed
metadata and references. PostgreSQL is operational metadata only; it is not the trajectory lake.

Do not add evaluators to the collector request path. Ingestion must remain bounded and deterministic;
evaluation and reward computation run in batch pipelines.

Commands: `make sync`, `make check`, `make demo`. Generated data belongs under `data/` or `outputs/`
and must not be committed.
