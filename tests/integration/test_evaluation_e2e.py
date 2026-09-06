from __future__ import annotations

import pytest

from refaat.application.runtime import Refaat
from refaat.domain.models import DocumentInput
from refaat.evaluation import Dataset, default_bundle

pytestmark = pytest.mark.integration


def test_program_evaluation_over_real_corpus(runtime: Refaat) -> None:
    project = runtime.project(workspace="acme", corpus="handbook")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="incident.txt",
            title="Incident",
            media_type="text/plain",
            content=b"Escalate a Sev1 incident to the on-call engineer within fifteen minutes of detection.",
        ),
    )
    dataset = Dataset.from_records(
        "incident-regression",
        [
            {"query": "incident escalation", "expected_terms": ["incident"]},
            {"query": "on-call response time", "expected_terms": ["minutes"]},
        ],
    )

    report = project.programs.get("grounded-rag").evaluate(dataset, default_bundle())

    assert report.program == "grounded-rag"
    assert report.case_count == 2
    assert report.hard_constraints_passed is True
    assert report.score > 0.0
    assert report.dataset_version.startswith("sha256:")
    assert report.to_json()["hard_constraints_passed"] is True
    assert len(report.cases) == 2


def test_durable_program_evaluation_job(runtime: Refaat) -> None:
    from refaat.application.worker import DurableWorker

    project = runtime.project(workspace="acme", corpus="handbook")
    runtime.ingest(
        project.workspace_id,
        project.corpus_id,
        DocumentInput(
            source_id="incident.txt",
            title="Incident",
            media_type="text/plain",
            content=b"Escalate a Sev1 incident to on-call within fifteen minutes of detection.",
        ),
    )
    dataset = Dataset.from_records("durable", [{"query": "incident escalation", "expected_terms": ["incident"]}])

    job_id = project.programs.get("grounded-rag").enqueue_evaluation(dataset)
    assert DurableWorker(runtime).run_once() is True

    job = runtime.store.get_job(project.workspace_id, job_id)
    assert job["status"] == "succeeded"
    assert job["result"]["program"] == "grounded-rag"
    assert job["result"]["hard_constraints_passed"] is True
