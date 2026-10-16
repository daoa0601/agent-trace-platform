# AgentTrace Platform

AgentTrace is a production-oriented data plane for collecting general agent traces, preserving them
immutably, rebuilding validated episodes, running versioned evaluations, and freezing reproducible
datasets for debugging, regression evaluation, SFT, preference optimization, or RL.

AgentTrace generalizes trace-collection ideas first prototyped in the frozen course repository
[`daoa0601/cs-e4645-data-intensive-computing`](https://github.com/daoa0601/cs-e4645-data-intensive-computing).

This checkout needs the sibling `benchmark-audit-core` next to it; see
[the catalog README](https://github.com/daoa0601/agentic-orch/blob/main/README.md#cross-repo-boundaries) for the layout and cross-repo boundaries.

## What is ready

- Native batched event ingestion with API-key authentication, request limits, retry-safe object
  names, secret/PII redaction, and content-addressed payload offloading.
- OTLP/HTTP ingestion through an OpenTelemetry Collector, including OpenInference span-kind mapping.
- A framework-neutral OpenInference SDK for agent, model, tool, evaluator, and environment spans,
  plus trace-correlated redacted JSON logging.
- An explicit run → episode → ordered-step schema with model, tool, environment, reward, privacy,
  token-level RL, and provenance fields.
- Immutable raw JSONL-Zstandard objects and checksummed canonical Parquet snapshots.
- Cross-event validation and quarantine for duplicate/conflicting IDs, broken refs, bad ordering,
  missing parents/tool results/terminal states, and inconsistent tokenizer versions.
- Domain-neutral, versioned evaluators and decomposed rewards.
- Frozen eval, SFT, preference, and strict-RL dataset builders with source snapshot manifests.
- Versioned task contracts, reproducible Docker environment setup, MinIO, PostgreSQL operational
  metadata, Prometheus alerts, and a provisioned Grafana dashboard.
- Provider-neutral JSONL samples, versioned task evaluators, and an optional Inspect AI adapter.
- Framework-neutral OpenTelemetry/OpenInference trace export, span-derived RED metrics, and optional
  Phoenix trace exploration.
- Explicit compile/check/benchmark/audit-retest stage spans and immutable, content-addressed
  storage for audit bundles built with `benchmark-audit-core`.

AgentTrace is the trace lake only. Benchmark execution, model-provider harnesses, and claim gating
are out of scope; the catalog README names the repository that owns them.

## Public Python API

Code outside this repository should import only these modules:

| Module | Use |
|---|---|
| `agenttrace.models` | canonical event, step, task-reference, and token-usage schemas |
| `agenttrace.sdk` | `TraceClient` / `TraceSession` for native batched trace emission |
| `agenttrace.integrations` | `create_telemetry`, `TelemetrySettings`, `CaptureMode`, `AgentRun`, `SpanResult` |
| `agenttrace.tasks` | `TaskDefinition`, `EvaluatorSpec`, `TaskSample`, loaders, prompt rendering, JSON digests, `parse_json_output`, content-free error descriptions |
| `agenttrace.storage` | `ObjectStore`, `FileObjectStore`, `S3ObjectStore`, `BlobStore`, `RawBatchWriter` |
| `agenttrace.redaction` | `Redactor`, `RedactionStats`, and the shared secret/private-reasoning rules |
| `agenttrace.audits` | `persist_audit_bundle(objects, bundle)` for immutable audit-bundle storage |
| `agenttrace.pipelines` | normalize, episode, evaluation, and dataset snapshot builders |

Other submodules are implementation details and may change without notice.

## Data flow

```text
Agent runtime / OpenTelemetry
            |
            v
      collector service  ----> Prometheus / Grafana
            |
            v
  immutable raw objects + content-addressed blobs
            |
            v
 canonical Parquet snapshot  ---> quarantine
            |
            v
 validated ordered episodes  ---> quarantine
            |
            v
 versioned evaluators + decomposed rewards
            |
            v
 frozen eval / SFT / preference / RL dataset
```

PostgreSQL stores task registrations, run lookup state, pipeline jobs, annotations, and dataset
registrations. It deliberately does **not** store the trajectory lake.

## Quick start

Requirements: Python 3.11, [`uv`](https://docs.astral.sh/uv/), and Docker only for the service stack.

```bash
uv sync --all-extras
uv run agenttrace task list --root taskpacks
uv run agenttrace demo --root outputs/demo
uv run pytest -v
```

The demo exercises the entire local path without network services and prints manifest keys for the
raw, canonical, episode, evaluation, and frozen dataset snapshots.

Run all quality gates:

```bash
make check
```

## Run the local service stack

```bash
cp .env.example .env
# Replace every development credential in .env.
docker compose up --build -d
curl http://localhost:8000/health/ready
```

Endpoints:

- native collector: `POST http://localhost:8000/v1/events`
- OTLP/gRPC: `localhost:4317`
- OTLP/HTTP: `http://localhost:4318`
- MinIO console: `http://localhost:9001`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

The Collector derives call counts, errors, and duration histograms from spans on port `8890`; those
series are scraped by Prometheus and shown in the provisioned dashboard. It deliberately excludes
run, episode, user, and prompt identifiers from metric labels.

Enable the optional Phoenix fan-out without changing the immutable AgentTrace sink:

```bash
docker compose -f compose.yaml -f compose.phoenix.yaml up --build -d
open http://localhost:6006
```

The Compose deployment is a reproducible local reference. Before shared or production deployment,
complete the controls in [the operations runbook](docs/operations.md) and
[security model](docs/security.md).

## Instrument an agent runtime

```python
from pathlib import Path

from agenttrace.models import Content, ModelInvocation, StepType, TaskRef
from agenttrace.sdk import TraceClient

with TraceClient(
    "http://localhost:8000",
    api_key="local-development-key",
    spool_dir=Path("outputs/trace-spool"),
) as client:
    with client.session(
        experiment_id="support-routing-v1",
        task=TaskRef(task_id="support.intent-routing", task_version="1.0.0"),
    ) as trace:
        trace.emit(StepType.USER_INPUT, input={"message": "I was charged twice."})
        with trace.step(StepType.MODEL_GENERATION) as step:
            step.model = ModelInvocation(provider="openai-compatible", model_id="policy-model")
            step.output = Content(inline={"intent": "billing_refund", "route": "billing"})
        trace.emit(StepType.FINAL_OUTPUT, output={"intent": "billing_refund", "route": "billing"})
```

For existing OpenTelemetry/OpenInference instrumentation, send OTLP to port 4317 or 4318. Add the
attributes `agenttrace.experiment.id`, `agenttrace.run.id`, `agenttrace.episode.id`, and especially
`agenttrace.sequence_number`; the adapter can infer ordering inside one export request, but explicit
sequence numbers remain the cross-worker source of truth.

For runtimes without an auto-instrumentor, use `agenttrace.integrations.telemetry`. It emits ordinary
OTLP spans with OpenInference kinds and adds task, sequence, provenance, usage, and correlation
attributes that the collector maps into canonical events. Content capture defaults to `metadata`,
which records byte counts and SHA-256 digests rather than prompts or responses.

For autonomous benchmark loops, use `run.benchmark_stage("compile" | "check" | "benchmark" |
"audit_retest")`. [Auditable benchmarks](docs/auditable-benchmarks.md) describes what AgentTrace
records and stores for an audited benchmark session.

Use `AGENTTRACE_OTLP_HEADERS='X-AgentTrace-Key=...'` when the destination OTLP endpoint itself
requires authentication. To export directly to Langfuse v4, set the OTLP endpoint to
`$LANGFUSE_BASE_URL/api/public/otel` and put a Basic authorization header plus
`x-langfuse-ingestion-version=4` in `AGENTTRACE_OTLP_HEADERS`. Keep that header in the environment,
not a command argument.

## Batch pipeline

Each command consumes a manifest from the previous stage and emits a new immutable manifest:

```bash
uv run agenttrace pipeline normalize --object-root data/object-store
uv run agenttrace pipeline episodes <canonical-manifest-key>
uv run agenttrace pipeline evaluate <episode-manifest-key> --task-root taskpacks
uv run agenttrace pipeline dataset my-eval eval <episode-manifest-key> <evaluation-manifest-key>
```

Use `eval`, `sft`, `preference`, or `rl` as the dataset kind. Strict RL exports reject episodes that
lack policy checkpoint, tokenizer, generation-engine, or behavior-logprob provenance.

## Repository layout

```text
src/agenttrace/       schemas, SDK, collector, integrations, storage, and pipelines
services/collector/   deployable collector image
pipelines/            batch-job boundary and deployment notes
taskpacks/            versioned task/evaluator/data-policy definitions
environments/         reproducible task sandbox images
infra/                MinIO/PostgreSQL/OTel/Prometheus/Grafana configuration
tests/                unit, API, quality, and end-to-end tests
docs/                 architecture, schema, security, and operations decisions
```

## Design constraints

- Raw objects are append-only. Corrections always produce another derived snapshot.
- Large values are referenced by SHA-256 digest instead of copied into analytical rows.
- Evaluators never run on the collector request path.
- Task, model, prompt, tokenizer, reward, environment, and source dataset versions are data.
- Training exports are immutable and honor each task's `allow_training` and private-reasoning policy.
- Object storage is the trace source of truth; PostgreSQL is the operational control plane.

The current implementation uses the recommended small-team stack—S3/MinIO, Parquet, DuckDB-compatible
tables, and PostgreSQL. At larger scale, place the canonical Parquet tables under Iceberg and run the
same logical stages with Spark, Ray, or Flink; the raw and snapshot contracts do not need to change.
