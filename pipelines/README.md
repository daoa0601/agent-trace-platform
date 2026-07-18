# Batch pipelines

The Python implementations live under `src/agenttrace/pipelines`; this directory documents the
deployable jobs and is the natural home for future Dagster, Airflow, Spark, or Ray entrypoints.

The supported dependency chain is:

1. `normalize`: raw JSONL-Zstandard objects to canonical Parquet tables.
2. `episodes`: ordered trajectories plus quality quarantine.
3. `evaluate`: versioned evaluator results and decomposed rewards.
4. `dataset`: immutable eval, SFT, preference, or RL snapshots.

Every stage consumes a manifest key and emits another immutable manifest. Re-running the same stage
with the same sources and configuration resolves to the same snapshot ID.
