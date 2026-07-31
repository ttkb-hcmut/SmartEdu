import asyncio
from importlib.util import find_spec


def test_process_tracer_module_exists():
    assert find_spec("knowledge.pipeline.tracer") is not None
    from knowledge.pipeline import tracer
    assert tracer.tracer_child.persist_result is True


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
    monkeypatch.setattr(tracer, "release_revision", lambda: "sha-123", raising=False)

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
