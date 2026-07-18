# Model benchmarking and trace collection

AgentTrace separates task semantics, model transport, telemetry, and evidence storage. A task and its
JSONL samples can therefore run through a direct provider API, an autonomous agent harness, or an
external evaluation framework without changing the retained result contract.

```text
TaskDefinition YAML + TaskSample JSONL
                  |
                  v
       one trace and run per sample
          /                    \
 general API transport      OpenCode transport
          \                    /
           OTLP spans + immutable evidence
                       |
                       v
       normalize -> episodes -> eval/datasets
```

The reusable layers are:

- `agenttrace.tasks`: versioned inputs, outputs, evaluator policy, privacy policy, and JSONL samples.
- `agenttrace.benchmarks`: provider-neutral result models, inline evaluators, and direct API runs.
- `agenttrace.harnesses.opencode`: isolated process/workspace lifecycle and Coding Plan credential
  boundary.
- `agenttrace.integrations.telemetry`: ordinary OpenTelemetry spans with OpenInference kinds.
- `agenttrace.storage` and `agenttrace.pipelines`: immutable evidence, normalized episodes, evaluations,
  and frozen downstream datasets.

Inspect AI is an optional adapter around the same tasks. It is useful for its model/eval ecosystem,
but neither the trace schema nor the stored benchmark record depends on it.

## Choose the credential path explicitly

| Credential | Command | Provider-key location | Endpoint |
|---|---|---|---|
| Z.AI general API | `benchmark zai-api` | parent process or mounted file | general API only |
| Z.AI Coding Plan | `benchmark zai-opencode` | parent process and loopback broker | Coding Plan only |
| AgentTrace collector | `AGENTTRACE_OTLP_HEADERS` | telemetry exporter only | configured OTLP sink |

There is no cross-plan fallback. The direct runner rejects Coding Plan selection, while the OpenCode
runner always uses the Coding Plan endpoint. This prevents an authentication failure from silently
changing quota or billing boundaries.

Z.AI's dedicated [GLM-5.2 guide](https://docs.z.ai/guides/llm/glm-5.2) documents the exact API
identifier `glm-5.2`, a 1M-token context window, and a 128K maximum output, so it is the default;
every command still accepts an explicit model. The
[core-parameter guide](https://docs.z.ai/guides/overview/concept-param) documents
`reasoning_effort` for GLM-5.2 and later. Coding Plan quota is restricted to supported coding tools
under Z.AI's [usage policy](https://docs.z.ai/devpack/usage-policy).
OpenCode is documented as a supported integration in Z.AI's
[tool guide](https://docs.z.ai/devpack/tool/others). Re-check those terms and model identifiers before
a live run; this project does not infer plan eligibility from the key.

## General API benchmark

```bash
export ZAI_API_KEY='...'
uv run agenttrace benchmark zai-api \
  taskpacks/examples/support-routing.yaml \
  taskpacks/examples/support-routing.samples.jsonl \
  --model glm-5.2 \
  --thinking-mode enabled \
  --reasoning-effort max \
  --metadata-only \
  --otlp-endpoint http://127.0.0.1:4318
```

Use `--api-key-file /mounted/secret` instead of the environment for a file-backed credential. The
runner creates one independently correlated trace for every sample, applies exact-match, containment,
regex, JSON Schema, and latency evaluators, and records provider errors without provider response
bodies. A required evaluator that needs a live environment or a custom registry fails validation
before model calls begin.

`agenttrace provider zai-smoke --endpoint-plan general` is the smallest authentication/connectivity
check. It prints only safe metadata and rejects Coding Plan use.

Direct calls default to `--thinking-mode disabled` for stable smoke tests. When thinking is enabled,
the runner sends and records `--reasoning-effort`; accepted values are `max`, `xhigh`, `high`,
`medium`, `low`, `minimal`, and `none`. Z.AI documents `max` as the default and notes that the setting
only takes effect while thinking is enabled.

## Coding Plan through OpenCode

Install and authenticate only through the parent-side credential source, then run:

```bash
export ZAI_API_KEY='...'
uv run agenttrace benchmark zai-opencode \
  taskpacks/examples/code-review.yaml \
  examples/inspect/code-review.jsonl \
  --workspace /absolute/path/to/frozen/source \
  --model glm-5.2 \
  --reasoning-effort max \
  --max-model-calls 64 \
  --watchdog-seconds 900 \
  --metadata-only \
  --otlp-endpoint http://127.0.0.1:4318
```

Use `--watchdog-seconds 0` for an intentionally unbounded session. Each sample receives:

1. A fresh copied workspace and a new OpenCode process.
2. An isolated `HOME` and XDG configuration/data/state tree.
3. A minimal OpenCode configuration with sharing, plugins, MCP servers, web tools, external-directory
   access, and auto-update disabled.
4. A short-lived local bearer capability usable only against a loopback broker.
5. A broker restricted to one model, the chat-completions path, a payload ceiling, and a maximum call
   count. It enforces the recorded thinking/reasoning policy, and injects the provider key only into
   the upstream request.

The source copy excludes `.git`, `.env*`, `.opencode`, `opencode.json*`, and `secrets`; a symlink that
escapes the source is rejected. The prompt lives in a mode-`0600` ephemeral config file and is never a
process argument. The child environment contains no known provider API-key variables.

These controls protect credentials and benchmark isolation; they are not an operating-system sandbox.
OpenCode can execute shell commands, and a host process may still have network and filesystem access
permitted by the OS. Run untrusted tasks inside a dedicated container or VM with a read-only base,
resource limits, no host credentials, and an explicit egress policy. OpenCode's own
[permissions](https://opencode.ai/docs/permissions) are an additional control, not that sandbox.

## Evidence and retention

The CLI prints a manifest key, never a prompt, response, or credential. Records and manifests are
content-addressed and immutable. Every OpenCode sample retains:

- the sanitized raw OpenCode JSONL session trace;
- hashes and byte counts for prompt, final response, stdout, and stderr;
- model/tool event counts, token usage, wall time, exit/timeout status, and redaction count;
- model-locked broker request status, size, and latency evidence without bodies or credentials;
- the requested model, thinking mode, and GLM-5.2 reasoning effort in traces, records, and manifests;
- content-addressed added/modified workspace files and deletion metadata;
- evaluator outcomes, aggregate score, task/sample/config/workspace digests, and trace identifiers.

`--metadata-only` omits the separate final-response object and keeps span payloads at hash/size level.
It deliberately does not remove the sanitized raw session JSONL because an end-to-end audit needs the
agent trajectory. Use only tasks whose data policy authorizes that retention. `--store-outputs` opts in
to a separately addressable final response; `--capture-content full` is an independent and more
sensitive opt-in for OTLP span content.

For a protected OTLP endpoint:

```bash
export AGENTTRACE_OTLP_HEADERS='X-AgentTrace-Key=collector-secret'
```

Keep this collector credential distinct from the model credential. The local OpenTelemetry Collector
on port `4318` normally injects its own downstream AgentTrace key, so clients do not need this header
when sending to that trusted local receiver.

## Metrics and audit workflow

The emitted spans support bounded-cardinality request/error/duration metrics by operation kind,
canonical step type, provider, and model. Run IDs, sample IDs, prompts, repository paths, and error text
remain trace attributes or immutable evidence—not Prometheus labels. Operationally monitor completed,
failed, and timed-out sessions; model/tool calls; input/output/cached/reasoning tokens; latency;
redaction volume; and collector rejection/queue health.

Treat any headline benchmark value as a claim rather than a row in a leaderboard. Freeze its task,
samples, source workspace, configuration, full sanitized trace, result files, and hardware observation
by digest. Then have a separate reviewer read the final artifact and full trace, run empirical re-tests
in a fresh worker/workspace, record suspected caching or grader games as findings, and gate publication
on resolved high-severity findings. The concrete manifest and CLI flow are in
[Auditable autonomous benchmarks](auditable-benchmarks.md).

## Add another provider or harness

A new model integration should implement only its transport and event adapter. Reuse `TaskSample`, the
benchmark record/evaluator models, `TelemetryRuntime`, and immutable object storage. Preserve one
trace/session per sample, explicit endpoint and credential configuration, safe provider errors, stable
digests, and no secrets in arguments, logs, traces, or manifests. Framework-specific event formats
belong under `harnesses/<name>` rather than in the task or storage schema.
