# Auditable autonomous benchmarks

AgentTrace treats a benchmark score as a claim linked to execution evidence, not as an unqualified
number. The protocol is domain-neutral: kernel optimization is the motivating example, but the same
contract works for browser agents, data-analysis agents, retrieval systems, and other tasks where an
agent can exploit the measurement setup.

## What is adopted

Each task instance runs in one isolated agent session on explicitly described live hardware. The
trace records compile, check, and benchmark stages (or the task's declared equivalents), including
failures and retries. The agent may decide when it is finished, while a watchdog remains mandatory
as an operational safety boundary. “Unlimited wall-clock” is therefore represented as
`termination_policy: agent_decides`, not as the absence of a timeout.

A headline claim requires an independent reviewer to inspect both the final artifact and the full
session trace. Audit schema `1.0.0` also requires completed manual review by a `human` or `hybrid`
auditor; an agent-only audit cannot satisfy that field. The reviewer performs fresh empirical
re-tests for cache sensitivity, input perturbations, output correctness, timing validity, and any
additional suspicious behavior. A high-severity unresolved finding blocks the claim. Findings are
preserved in a separate report section even when the overall verdict passes.

## Evidence chain

```text
versioned task + clean commit + hardware observation
                         |
                         v
             one autonomous session
          compile -> check -> benchmark
                         |
                         v
           full trace + final artifact
             + metric evidence digests
                         |
                         v
       independent code/trace review + re-tests
                         |
                         v
             deterministic headline gate
                         |
                         v
       immutable audit bundle + findings report
```

The standalone `benchmark-audit-core` package supplies strict Pydantic models and the deterministic
gate for this chain. `agenttrace.audits` re-exports the core model and gate symbols for
compatibility; the AgentTrace-owned addition is `persist_audit_bundle` and its object-store adapter.
All evidence uses a URI, SHA-256 digest, media type, and optional byte size. The hardware profile
records accelerator vendor/model/count, optimization scope, driver/runtime versions, and a digest
of observed inventory output. Do not put serial numbers, usernames, or other host identifiers in
manifests.

`benchmark-audit-core` is a workspace-local dependency by design. AgentTrace's
`[tool.uv.sources]` entry points to `../benchmark-audit-core`, so a relocated checkout or CI job
must supply that sibling source, vendor it, or use an explicitly pinned Git source. Any source
change must be followed by lock regeneration and frozen-environment verification.

## Trace instrumentation

Use explicit stage types so the audit gate can distinguish the live loop from ordinary model/tool
activity:

```python
with telemetry.run(
    experiment_id="kernel-study-v1",
    task_id="systems.kernel-optimization",
    task_version="1.0.0",
    attributes={
        "agenttrace.provenance.benchmark_protocol_version": "kernel-v1",
        "agenttrace.hardware.accelerator.vendor": "NVIDIA",
        "agenttrace.hardware.accelerator.model": "H100",
        "agenttrace.hardware.accelerator.count": 1,
        "agenttrace.hardware.optimization_scope": "single_device",
    },
) as run:
    with run.benchmark_stage("compile", input={"source_sha256": source_digest}) as stage:
        stage.output = {"exit_code": 0, "log_sha256": compile_log_digest}
    with run.benchmark_stage("check") as stage:
        stage.output = {"passed": True, "suite_version": "hidden-v3"}
    with run.benchmark_stage("benchmark") as stage:
        stage.output = {"samples_ref": samples_uri, "samples_sha256": samples_digest}
```

The root attributes are preserved in the canonical trace. High-cardinality hardware and run fields
stay out of Prometheus labels; they remain available in traces and immutable manifests.

## Independent audit workflow

1. Freeze the final artifact, full trace, raw benchmark samples, hardware inventory, task definition,
   source commit, and benchmark protocol by digest.
2. Create an `AuditableSession` manifest. Mark a result `headline: true` only if it should pass the
   stricter publication gate.
3. Give a separate reviewer the artifact and full trace. The reviewer must use a fresh process and,
   when feasible, a fresh workspace. Re-tests record their own evidence rather than referring only
   to the original score.
4. Attach the `IndependentAudit`, including all findings. `accepted` is not equivalent to `resolved`
   for critical or high findings and still blocks the headline gate.
5. Validate, render, and persist the bundle:

   ```bash
   uv run agenttrace audit gate path/to/session.json \
     --report outputs/audits/session.md \
     --object-root data/object-store
   ```

The command exits with code 2 when a headline is blocked, code 1 for an invalid manifest, and code 0
for a passed or non-applicable gate. Its JSON output contains only identifiers and gate status. The
content-addressed bundle is written under `audits/bundles/` without overwriting an existing object.
The validation and report are produced by `benchmark-audit-core`; persistence under that key remains
an AgentTrace responsibility.

## Default anti-gaming re-tests

- `fresh_process`: rebuild and rerun outside the generating agent process.
- `cold_cache`: invalidate or bypass relevant compilation/data caches and compare behavior.
- `input_perturbation`: change shapes, values, seeds, or task order within the declared domain.
- `output_validation`: compare complete outputs against an independent oracle or hidden suite.

Kernel tasks should normally add execution-order randomization, hidden checks, repeated measurement,
timing sanity, and resource-isolation tests, as shown in
`taskpacks/examples/kernel-optimization.yaml`. Other domains can select a different versioned set in
their task definition. The gate never infers a pass from missing evidence.

## Deliberate boundaries

- `benchmark-audit-core` validates evidence metadata; it does not retrieve referenced bytes, prove
  that a digest is authentic or relevant to a claim, execute re-tests, redact content, or enforce
  retention. Harnesses and AgentTrace integrations own those controls.
- AgentTrace stores traces and evidence; it does not route model requests, own provider retries or
  fallbacks, or calculate provider/gateway billing. There is no live Vercel AI Gateway integration
  in this repository.
- If a harness later uses Nexus or a managed gateway, exactly one layer must own retry/fallback. The
  harness must emit the resolved route and attempt history; AgentTrace can preserve those facts but
  cannot reconstruct hidden gateway attempts after the fact.
- The collector records evidence; it does not decide benchmark correctness on the ingestion path.
- A task-specific harness owns compilation, oracle checks, cache controls, and timing methodology.
- An agent cannot self-attest independence, and changing reviewer identity alone is insufficient if
  the audit reuses the original process or cache state.
- Historical claims are not upgraded merely because this protocol now exists. They need a new audit
  manifest and empirical re-test evidence.
