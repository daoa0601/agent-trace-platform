# Task environments

`EnvironmentSpec` supports no-op, Docker, Kubernetes, and remote contracts. The implemented local
adapter uses Docker without a shell, drops Linux capabilities, enables `no-new-privileges`, limits
CPU/memory/PIDs, disables networking by default, and mounts only a task workspace as writable.

Build the small reference image for local adapter tests:

```bash
docker build -t agenttrace/sandbox:0.1.0 environments/sandbox
docker image inspect agenttrace/sandbox:0.1.0 --format '{{index .RepoDigests 0}}'
```

Shared or production tasks must record the resulting `sha256:` image digest in their task version.
Kubernetes/remote adapters are intentionally contracts only until their isolation and snapshot
semantics can match the Docker adapter; runners must fail closed rather than silently use the host.
