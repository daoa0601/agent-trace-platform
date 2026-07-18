# Migration from the research project

Source reviewed: `daoa0601/cs-e4645-data-intensive-computing` at commit
`44e9f43cce92abe8e7d957257aa7c3a895d085ca` (2026-04-17).

The source remains a finalized research artifact. AgentTrace is a clean sibling implementation so the
report, evidence paths, LUMI scripts, archived baselines, and coding benchmark stay reproducible. No
production package imports the research tree, and no compatibility edits are required there.

| Research concept | Production replacement |
|---|---|
| `analysis/eval_trace.py` append-only local JSONL | OpenInference SDK + authenticated collector + immutable object batches |
| `analysis/experiment_record.py` | canonical run/episode/event schemas plus operational PostgreSQL tables |
| `custom_repo/issue_pack.py` and PRD manifests | domain-neutral `TaskDefinition` YAML plus `TaskSample` JSONL contracts |
| Symphony `raw_events.json` / `normalized_trace.json` | raw, canonical, and episode snapshot pipelines |
| coding-specific acceptance checks | versioned evaluator registry driven by task definitions |
| training scripts consuming ad hoc trajectories | frozen eval/SFT/preference/RL builders with manifests |
| local GPU/runtime monitoring | trace-correlated resource records plus spanmetrics, Prometheus alerts, and Grafana |

New collection and benchmarks run entirely from this sibling package. The general API runner and
OpenCode harness each emit experiment, run, episode, trace, and configuration-digest identifiers into
the same provider-neutral evidence contract. Legacy JSONL remains untouched for historical report
reproduction rather than becoming a production ingestion dependency.

The important research ideas were preserved: explicit experiment records, frozen tasks/environments,
traceable orchestration, acceptance gates, trajectory metrics, artifact verification, and training-data
provenance. Coding-only assumptions—repository URLs, patches, tests, SWE-Bench messages, and fixed PRD
features—are now optional task-domain data rather than core platform fields.
