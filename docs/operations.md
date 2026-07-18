# Operations runbook

## Local startup

1. Copy `.env.example` to `.env`, replace every credential, and override any occupied host ports.
2. Run `docker compose config` and `docker compose up --build -d`.
3. Check `/health/ready`, the OTel health port `13133`, MinIO, all three Prometheus targets, and Grafana.
4. Send a small native or OTLP trace and verify a `raw/collection_date=...` object appears.
5. Run the normalize → episodes → evaluate → dataset smoke path before accepting workload traffic.

## SLO signals

Alert on collector readiness, raw-write errors, queue saturation, retry age, request rejection, p95 raw
write latency, duplicate rate, and redaction volume changes. Batch jobs additionally report invalid
trace rate, episode completion, missing tool results, reward distribution, truncation, task/model mix,
and dataset exclusions in their manifests.

Audit jobs should additionally count passed, blocked, and non-applicable headline gates plus failed
check IDs. Check IDs are bounded schema values and safe as metric labels; session, claim, reviewer,
artifact, hardware-serial, and finding text are not. A blocked gate is a research-integrity outcome,
not a collector availability incident.

The spanmetrics connector exposes bounded-cardinality operation count and duration series. Keep its
dimensions limited to operation kind, canonical step type, provider, and model. Never add trace/run,
task instance, prompt, user, tool-call, repository, or error-message values as Prometheus labels.

Phoenix is an optional debugging projection, not a source of truth. Start it with the Compose override,
keep a finite retention policy, and verify that failure of Phoenix does not block the AgentTrace export.
The reference Phoenix container has no authentication and is suitable only on a trusted local host;
put authentication and TLS in front of any shared deployment.

## Backups and recovery

- Raw/blob buckets are primary data: enable versioning, cross-region replication as required, and test
  restores. Never repair an object in place.
- Back up PostgreSQL with PITR; it can be reconstructed partially from manifests, but annotations and
  job/control-plane state cannot.
- Grafana and Prometheus data are operationally useful but not trace sources of truth.
- Phoenix data is rebuildable/debug-only and is not included in dataset provenance.
- Preserve task definition versions and container digests for at least as long as derived datasets.
- Retain audit bundles and every referenced trace, artifact, protocol, hardware observation, and raw
  metric sample for at least as long as the associated claim is published. Version and replicate the
  evidence objects; a manifest whose references have expired is no longer independently reproducible.

## Deployment hardening

- Run multiple stateless collector replicas behind a load balancer.
- Use S3 conditional writes and a persistent OTel sending queue.
- Keep collector CPU/memory and request limits explicit; scale on queue length and write latency.
- Schedule batch compaction so Parquet files are hundreds of MB rather than tiny per-request files.
- Move canonical tables to Iceberg when incremental updates, partition evolution, or time travel across
  large volumes become necessary.
- Run evaluators in a separate trust domain; evaluator failures must not impact ingestion.
- Run empirical audit re-tests in a fresh worker identity and process/workspace, without mounting the
  generating session's writable cache. Reviewer jobs may read immutable evidence but must write their
  results under a separate audit prefix.

## Schema changes

Add fields compatibly within a schema version. A breaking semantic or type change gets a new version
and a normalizer capable of reading old raw events. Never rewrite old raw batches or frozen datasets.
