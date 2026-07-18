# Security and data governance

Agent traces routinely contain credentials, personal data, private documents, proprietary code, and
model-internal reasoning. Treat the trace bucket as restricted production data.

## Implemented controls

- Collector API keys are compared in constant time and are mandatory in production mode.
- Request bodies and native event batches are bounded.
- Unknown schema fields are rejected.
- Secret-bearing keys and common credential patterns are redacted at collection. Strict mode also
  redacts email addresses and phone-like values.
- Common private-reasoning keys are removed unless collection redaction is explicitly disabled.
- Manual instrumentation defaults to metadata-only content capture; Z.AI commands print only safe
  summaries and immutable manifest keys.
- The OpenCode transport gives the child only a short-lived loopback capability. Its model-locked,
  call-bounded broker keeps the Coding Plan key in the parent and records body-free request evidence.
- Every OpenCode sample receives an isolated configuration tree and copied workspace; `.env*`,
  `.opencode`, OpenCode configs, secrets directories, and escaping symlinks are excluded.
- Large content uses SHA-256-addressed immutable objects and is verified when read.
- Task data policy controls retention metadata, training authorization, and private-reasoning use.
- The Docker environment drops capabilities, sets `no-new-privileges`, limits resources, disables
  networking by default, and does not invoke a shell.

## Required production controls

The reference Compose stack is not an internet-facing deployment. A shared deployment must add:

1. TLS/mTLS at ingress and OTLP endpoints; rotate collector credentials from a secret manager.
2. Cloud IAM or workload identity instead of static S3 credentials.
3. Bucket encryption with customer-managed keys, versioning/Object Lock where required, access logs,
   lifecycle rules, and separate restricted/raw versus broadly queryable canonical roles.
4. Tenant-scoped authorization and object prefixes if more than one trust boundary uses the service.
5. Network policies around the collector, object store, control plane, and task sandboxes.
6. A reviewed retention/deletion workflow for user data and legal holds.
7. Audit logging for task changes, dataset registration, annotation, export, and blob access.
8. Image signing/scanning and digest allowlists for task environments.

Redaction is defense in depth, not proof that a trace is safe. Do not collect content you are not
authorized to retain, and never infer training permission merely from internal access.

OpenCode process/configuration isolation is not an operating-system sandbox. For untrusted agent tasks,
run the entire harness in a dedicated container or VM with no host secrets, a read-only base, explicit
resource limits, and restricted egress. Shell commands otherwise retain the permissions of the harness
user even when OpenCode's built-in web and external-directory tools are denied.

Provider credentials must come from process environment, workload identity, or a mounted secret file.
Do not accept keys as CLI flags, serialize `SecretStr` settings, put credentials in `.env.example`, or
copy them into trace/log attributes. Endpoint-plan selection is also a security and billing boundary:
never fall back between Z.AI general API and Coding Plan endpoints automatically.
