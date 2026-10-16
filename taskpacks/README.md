# Task packs

Tasks are versioned data contracts, not runner-specific scripts. Each YAML file declares input and
output schemas, environment isolation, evaluator versions and weights, privacy/retention policy,
and whether the resulting episodes may be used for training. Benchmark tasks may additionally
declare session and independent-audit policies. The kernel example uses custom correctness and
timing evaluators because those checks belong to the concrete benchmark harness, not the collector.

Concrete cases live in strict `TaskSample` JSONL files so the same task/sample pair can run through a
direct model API, OpenCode, Inspect AI, or another adapter. Each row has a stable `sample_id`, JSON
`input`, optional `target`, and low-sensitivity metadata. Keep secrets and mutable environment paths out
of both YAML and JSONL; resolve them only in the selected harness.

Validate all task definitions with:

```bash
uv run agenttrace task list --root taskpacks
uv run agenttrace task validate taskpacks/examples/support-routing.yaml
uv run agenttrace task validate taskpacks/examples/kernel-optimization.yaml
```

Changing a prompt, evaluator, reference answer, container image, or data policy requires a new task
version. Historical traces continue to resolve against the exact version they recorded.

Task policy is necessary but not sufficient for a publishable benchmark claim. Each actual run must
also produce an `AuditableSession` manifest with observed hardware evidence, content digests, stage
counts, and an independent audit. The portable policy types come from `benchmark-audit-core`;
AgentTrace embeds them in task definitions and persists audit bundles. See
[the benchmark integrity protocol](../docs/auditable-benchmarks.md).
