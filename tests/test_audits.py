from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from agenttrace.audits import (
    AuditableSession,
    AuditFinding,
    AuditorKind,
    AuditPolicy,
    AuditVerdict,
    BenchmarkClaim,
    BenchmarkStage,
    EmpiricalRetest,
    EvidenceRef,
    FindingCategory,
    FindingSeverity,
    FindingStatus,
    GateStatus,
    HardwareProfile,
    IndependentAudit,
    OptimizationScope,
    RetestCategory,
    RetestOutcome,
    TerminationReason,
    build_audit_bundle,
    evaluate_headline_gate,
    persist_audit_bundle,
    render_audit_report,
)
from agenttrace.storage import FileObjectStore
from benchmark_audit_core import AuditableSession as CoreAuditableSession
from benchmark_audit_core import build_audit_bundle as core_build_audit_bundle


def test_agenttrace_audit_imports_reexport_core_objects() -> None:
    assert AuditableSession is CoreAuditableSession
    assert build_audit_bundle is core_build_audit_bundle


def _evidence(name: str) -> EvidenceRef:
    digest = {"trace": "1", "artifact": "2", "hardware": "3", "claim": "4"}.get(name, "5")
    return EvidenceRef(
        uri=f"object://evidence/{name}",
        sha256=digest * 64,
        media_type="application/json",
        byte_size=100,
    )


def _audit(ended_at: datetime) -> IndependentAudit:
    retests = [
        EmpiricalRetest(
            retest_id=f"retest-{category.value}",
            category=category,
            outcome=RetestOutcome.PASSED,
            isolated_from_original_session=True,
            evidence=[_evidence(f"retest-{category.value}")],
        )
        for category in (
            RetestCategory.FRESH_PROCESS,
            RetestCategory.COLD_CACHE,
            RetestCategory.INPUT_PERTURBATION,
            RetestCategory.OUTPUT_VALIDATION,
        )
    ]
    return IndependentAudit(
        audit_id="audit-001",
        auditor_id="review-agent-02",
        auditor_kind=AuditorKind.HYBRID,
        independent=True,
        manual_review_completed=True,
        reviewed_final_artifact=True,
        reviewed_full_trace=True,
        reviewed_task_definition=True,
        reviewed_benchmark_protocol=True,
        retests=retests,
        verdict=AuditVerdict.PASS,
        completed_at=ended_at + timedelta(minutes=30),
    )


def _session(*, headline: bool = True, source_dirty: bool = False) -> AuditableSession:
    started_at = datetime(2026, 7, 17, 8, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(hours=2)
    return AuditableSession(
        session_id="session-001",
        experiment_id="kernel-study-v1",
        run_id="run-001",
        episode_id="episode-001",
        trace_id="a" * 32,
        task_id="systems.kernel-optimization",
        task_version="1.0.0",
        generator_id="optimizer-agent-01",
        source_commit="abcdef1234567890",
        source_dirty=source_dirty,
        benchmark_protocol_version="kernel-v1",
        task_definition=_evidence("task-definition"),
        benchmark_protocol=_evidence("benchmark-protocol"),
        started_at=started_at,
        ended_at=ended_at,
        termination_reason=TerminationReason.AGENT_FINISHED,
        hardware=HardwareProfile(
            accelerator_vendor="NVIDIA",
            accelerator_models=["H100"],
            accelerator_count=1,
            optimization_scope=OptimizationScope.SINGLE_DEVICE,
            host_architecture="x86_64",
            driver_version="test-driver",
            runtime_versions={"cuda": "test-runtime"},
            observed_evidence=_evidence("hardware"),
        ),
        stage_counts={
            BenchmarkStage.COMPILE: 4,
            BenchmarkStage.CHECK: 4,
            BenchmarkStage.BENCHMARK: 12,
        },
        benchmark_repetitions=10,
        full_trace=_evidence("trace"),
        final_artifact=_evidence("artifact"),
        claims=[
            BenchmarkClaim(
                claim_id="latency-main",
                metric_name="median latency",
                value=12.5,
                unit="microseconds",
                headline=headline,
                evidence=[_evidence("claim")],
            )
        ],
        audit=_audit(ended_at),
    )


def test_valid_audited_session_passes_and_is_deterministic(tmp_path: Path) -> None:
    session = _session()
    first = build_audit_bundle(session)
    second = build_audit_bundle(session)

    assert first.headline_gate.status == GateStatus.PASSED
    assert first.headline_gate.eligible_for_headline is True
    assert first.digest() == second.digest()
    assert session.digest() == first.headline_gate.session_digest

    report = render_audit_report(first)
    assert "## Empirical re-tests" in report
    assert "## Audit findings" in report
    assert "No findings were recorded." in report

    store = FileObjectStore(tmp_path)
    created = persist_audit_bundle(store, first)
    duplicate = persist_audit_bundle(store, second)
    assert created.created is True
    assert duplicate.created is False
    assert store.get_bytes(created.key) == first.canonical_bytes()


def test_headline_gate_blocks_dirty_source_and_unresolved_high_finding() -> None:
    session = _session(source_dirty=True)
    assert session.audit is not None
    session.audit.findings.append(
        AuditFinding(
            finding_id="cache-001",
            category=FindingCategory.CACHE_CONTAMINATION,
            severity=FindingSeverity.HIGH,
            status=FindingStatus.ACCEPTED,
            observation="The benchmark reused a generated cache entry.",
        )
    )

    result = evaluate_headline_gate(session)
    failed = {check.check_id for check in result.checks if not check.passed}
    assert result.status == GateStatus.BLOCKED
    assert "source.clean" in failed
    assert "audit.blocking_findings" in failed


def test_missing_or_non_isolated_retest_blocks_headline() -> None:
    session = _session()
    assert session.audit is not None
    session.audit.retests = [
        retest for retest in session.audit.retests if retest.category != RetestCategory.COLD_CACHE
    ]

    result = evaluate_headline_gate(session)
    failed = {check.check_id for check in result.checks if not check.passed}
    assert "retest.cold_cache" in failed


def test_failed_additional_retest_cannot_be_hidden_by_passing_defaults() -> None:
    session = _session()
    assert session.audit is not None
    session.audit.retests.append(
        EmpiricalRetest(
            retest_id="retest-timing-sanity",
            category=RetestCategory.TIMING_SANITY,
            outcome=RetestOutcome.FAILED,
            isolated_from_original_session=True,
            evidence=[_evidence("timing-sanity")],
        )
    )

    result = evaluate_headline_gate(session)
    failed = {check.check_id for check in result.checks if not check.passed}
    assert "audit.retest_outcomes" in failed


def test_non_headline_claim_is_not_subject_to_publication_gate() -> None:
    result = evaluate_headline_gate(_session(headline=False))
    assert result.status == GateStatus.NOT_APPLICABLE
    assert result.eligible_for_headline is False
    assert result.checks == []


def test_single_device_profile_and_independent_identity_are_validated() -> None:
    with pytest.raises(ValidationError, match="exactly one accelerator"):
        HardwareProfile(
            accelerator_vendor="NVIDIA",
            accelerator_models=["H100", "H100"],
            accelerator_count=2,
            optimization_scope=OptimizationScope.SINGLE_DEVICE,
            host_architecture="x86_64",
            observed_evidence=_evidence("hardware"),
        )

    session = _session()
    assert session.audit is not None
    payload = session.model_dump(mode="json")
    payload["audit"]["auditor_id"] = session.generator_id
    with pytest.raises(ValidationError, match="independent auditor"):
        AuditableSession.model_validate(payload)

    with pytest.raises(ValidationError, match="cannot remove baseline"):
        AuditPolicy(required_retests=[])
    with pytest.raises(ValidationError, match="must remain blocking"):
        AuditPolicy(blocking_severities=[FindingSeverity.CRITICAL])


def test_audit_cli_renders_and_persists_passed_bundle(tmp_path: Path) -> None:
    from agenttrace.cli import app

    manifest = tmp_path / "session.json"
    report = tmp_path / "report.md"
    object_root = tmp_path / "objects"
    manifest.write_text(_session().model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "audit",
            "gate",
            str(manifest),
            "--report",
            str(report),
            "--object-root",
            str(object_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"status": "passed"' in result.output
    assert "## Audit findings" in report.read_text(encoding="utf-8")
    assert len(list(object_root.glob("audits/bundles/*.json"))) == 1
