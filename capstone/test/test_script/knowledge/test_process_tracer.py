import asyncio
from importlib.util import find_spec

import pytest


def test_process_tracer_module_exists():
    assert find_spec("knowledge.pipeline.tracer") is not None
    from knowledge.pipeline import tracer
    assert tracer.tracer_child.persist_result is True


def test_tracer_child_reports_its_release_revision(monkeypatch):
    from knowledge.pipeline import tracer

    monkeypatch.setattr(tracer, "release_revision", lambda required=False: "sha-123")
    result = tracer.tracer_child.fn("proof")

    assert result["token"] == "proof"
    assert result["release_revision"] == "sha-123"


def test_tracer_root_waits_for_the_child_process_result(monkeypatch):
    from knowledge.pipeline import tracer

    calls = []

    class _State:
        def result(self, raise_on_failure):
            assert raise_on_failure is True
            return {"token": "proof", "pid": 42}

    class _Run:
        state = _State()

    monkeypatch.setattr(
        tracer,
        "run_deployment",
        lambda **kwargs: calls.append(kwargs) or _Run(),
        raising=False,
    )

    assert asyncio.run(tracer.tracer_root.fn("proof")) == {"token": "proof", "pid": 42}
    assert calls == [{
        "name": "ingest-process-tracer-child/local",
        "parameters": {"token": "proof"},
        "timeout": None,
        "as_subflow": True,
    }]


def test_deploy_tracer_uses_the_asr_process_pool(monkeypatch, tmp_path):
    from knowledge.pipeline import tracer

    deployments = []

    class _Source:
        async def deploy(self, **kwargs):
            deployments.append(kwargs)
            return "tracer-deployment"

    class _Flow:
        async def from_source(self, source, entrypoint):
            assert source == tmp_path
            assert entrypoint == "knowledge/pipeline/tracer.py:tracer_child"
            return _Source()

    monkeypatch.setattr(tracer, "tracer_child", _Flow())
    monkeypatch.setattr(
        tracer, "release_revision", lambda required=False: "sha-123", raising=False
    )

    assert asyncio.run(tracer.deploy_tracer(tmp_path)) == "tracer-deployment"
    assert deployments == [{
        "name": "local",
        "work_pool_name": "ingest-asr",
        "build": False,
        "push": False,
        "job_variables": {"working_dir": str(tmp_path)},
        "version": "sha-123",
        "ignore_warnings": True,
        "print_next_steps": False,
    }]


def test_tracer_run_rejects_a_worker_on_another_release(monkeypatch):
    from knowledge.pipeline import tracer

    async def deployed():
        return None

    async def traced(token):
        return {
            "token": token,
            "pid": tracer.os.getpid() + 1,
            "release_revision": "sha-worker",
        }

    monkeypatch.setattr(tracer, "release_revision", lambda required=False: "sha-root")
    monkeypatch.setattr(tracer, "deploy_tracer", deployed)
    monkeypatch.setattr(tracer, "tracer_root", traced)

    with pytest.raises(RuntimeError, match="release revision differs"):
        asyncio.run(tracer.run())
