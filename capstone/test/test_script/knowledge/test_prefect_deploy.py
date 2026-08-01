import asyncio
from importlib.util import find_spec
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]


def test_stage_deployment_module_exists():
    assert find_spec("knowledge.pipeline.deploy") is not None


def test_stage_deployments_have_one_capability_pool_each():
    from knowledge.pipeline import deploy

    assert deploy.POOL_NAMES == ("ingest-ocr", "ingest-llm", "ingest-asr")
    assert {
        name: pool
        for name, (_, pool, _) in deploy.STAGES.items()
    } == {
        "ocr-slide": "ingest-ocr",
        "ocr-textbook": "ingest-ocr",
        "llm-slide": "ingest-llm",
        "asr-video": "ingest-asr",
    }
    assert {
        name: entrypoint
        for name, (_, _, entrypoint) in deploy.STAGES.items()
    } == {
        "ocr-slide": "knowledge/pipeline/flows.py:ocr_slide_stage",
        "ocr-textbook": "knowledge/pipeline/flows.py:ocr_textbook_stage",
        "llm-slide": "knowledge/pipeline/flows.py:llm_slide_stage",
        "asr-video": "knowledge/pipeline/flows.py:asr_video_stage",
    }


def test_deploy_stages_uses_checkout_without_building_or_pushing(monkeypatch, tmp_path):
    from knowledge.pipeline import deploy

    sources, deployments = [], []

    class _Source:
        def deploy(self, **kwargs):
            deployments.append(kwargs)
            return kwargs["name"]

    class _Flow:
        async def from_source(self, source, entrypoint):
            sources.append((source, entrypoint))
            return _Source()

    monkeypatch.setattr(
        deploy,
        "STAGES",
        {
            "ocr-slide": (_Flow(), "ingest-ocr", "knowledge/pipeline/flows.py:ocr_slide_stage"),
            "ocr-textbook": (_Flow(), "ingest-ocr", "knowledge/pipeline/flows.py:ocr_textbook_stage"),
            "llm-slide": (_Flow(), "ingest-llm", "knowledge/pipeline/flows.py:llm_slide_stage"),
            "asr-video": (_Flow(), "ingest-asr", "knowledge/pipeline/flows.py:asr_video_stage"),
        },
    )
    monkeypatch.setattr(deploy, "release_revision", lambda required=False: "sha-123")

    assert asyncio.run(deploy.deploy_stages(tmp_path)) == [
        "ocr-slide", "ocr-textbook", "llm-slide", "asr-video"
    ]
    assert sources == [
        (tmp_path, "knowledge/pipeline/flows.py:ocr_slide_stage"),
        (tmp_path, "knowledge/pipeline/flows.py:ocr_textbook_stage"),
        (tmp_path, "knowledge/pipeline/flows.py:llm_slide_stage"),
        (tmp_path, "knowledge/pipeline/flows.py:asr_video_stage"),
    ]
    assert [item["work_pool_name"] for item in deployments] == [
        "ingest-ocr", "ingest-ocr", "ingest-llm", "ingest-asr"
    ]
    for item in deployments:
        assert item["build"] is False
        assert item["push"] is False
        assert item["version"] == "sha-123"
        assert item["job_variables"] == {"working_dir": str(tmp_path)}
        assert item["ignore_warnings"] is True


def test_deploy_stages_rejects_an_unidentified_release(monkeypatch, tmp_path):
    from knowledge.pipeline import deploy

    def unidentified(required=False):
        assert required is True
        raise RuntimeError("INGEST_RELEASE_REVISION must identify the deployed Git revision")

    monkeypatch.setattr(deploy, "release_revision", unidentified)

    with pytest.raises(RuntimeError, match="INGEST_RELEASE_REVISION"):
        asyncio.run(deploy.deploy_stages(tmp_path))


def test_create_process_pools_is_repeatable(monkeypatch):
    from knowledge.pipeline import deploy

    created = []

    class _Client:
        async def create_work_pool(self, work_pool, overwrite):
            created.append((work_pool.name, work_pool.type, overwrite))
            return work_pool.name

    class _ClientContext:
        async def __aenter__(self):
            return _Client()

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(deploy, "get_client", lambda: _ClientContext(), raising=False)

    assert asyncio.run(deploy.create_process_pools()) == [
        "ingest-ocr", "ingest-llm", "ingest-asr"
    ]
    assert created == [
        ("ingest-ocr", "process", True),
        ("ingest-llm", "process", True),
        ("ingest-asr", "process", True),
    ]


def test_register_creates_pools_before_stage_deployments(monkeypatch, tmp_path):
    from knowledge.pipeline import deploy

    calls = []

    async def create_pools():
        calls.append("pools")
        return ["ingest-ocr", "ingest-llm", "ingest-asr"]

    async def deploy_stages(checkout):
        calls.append(("deploy", checkout))
        return ["ocr-slide", "ocr-textbook", "llm-slide", "asr-video"]

    monkeypatch.setattr(deploy, "create_process_pools", create_pools)
    monkeypatch.setattr(deploy, "deploy_stages", deploy_stages)

    assert asyncio.run(deploy.register(tmp_path)) == {
        "pools": ["ingest-ocr", "ingest-llm", "ingest-asr"],
        "deployments": ["ocr-slide", "ocr-textbook", "llm-slide", "asr-video"],
    }
    assert calls == ["pools", ("deploy", tmp_path)]


def test_process_pool_operations_are_documented_in_adr_0009():
    adr = ROOT / "docs" / "adr" / "0009-capability-routed-process-pools.md"
    assert adr.is_file()

    text = adr.read_text(encoding="utf-8")
    assert "ADR-0007" in text
    assert "ingest-ocr" in text
    assert "ingest-llm" in text
    assert "ingest-asr" in text
    assert "No Consul" in text

    readme = (ROOT / "infra" / "prefect" / "README.md").read_text(encoding="utf-8")
    assert "INGEST_CHECKOUT" in readme
    assert "INGEST_RELEASE_REVISION" in readme
    assert "uv run python -m knowledge.pipeline.deploy" in readme
    for pool in ("ingest-ocr", "ingest-llm", "ingest-asr"):
        assert f"prefect worker start --pool {pool} --type process --limit 1" in readme
