import runpy
import sys
from pathlib import Path
from types import ModuleType

import yaml
from prefect.client.schemas.objects import ConcurrencyLimitStrategy


ROOT = Path(__file__).resolve().parents[4]
INFRA = ROOT / "infra" / "prefect"


def test_prefect_infra_is_independent_private_and_pinned():
    compose = yaml.safe_load((INFRA / "compose.yml").read_text(encoding="utf-8"))
    svc = compose["services"]

    assert set(svc) == {"postgres", "prefect-server", "sftp"}
    assert svc["postgres"]["image"].startswith("postgres:14.")
    assert svc["prefect-server"]["image"] == "prefecthq/prefect:3.7.7-python3.12"
    assert svc["postgres"].get("ports") is None
    assert svc["prefect-server"]["ports"][0].startswith("${PREFECT_BIND_ADDR")
    assert svc["sftp"]["ports"][0].startswith("${PREFECT_BIND_ADDR")
    assert compose["networks"]["prefect-private"]["internal"] is True


def test_result_store_uses_private_sftp_and_worker_lock():
    cfg = yaml.safe_load((INFRA / "result-store.yml").read_text(encoding="utf-8"))
    req = (INFRA / "requirements.txt").read_text(encoding="utf-8")

    assert cfg["block"]["type"] == "remote-file-system"
    assert cfg["block"]["name"] == "prefect-sftp-results"
    assert cfg["block"]["basepath"].startswith("sftp://")
    assert cfg["block"]["settings"]["port"] == 2222
    assert "prefect==3.7.7" in req
    assert "paramiko" in req


def test_serve_registers_one_global_enqueue_slot(monkeypatch):
    calls = []

    def fake_serve(**kwargs):
        calls.append(kwargs)

    fake = ModuleType("knowledge.pipeline.flows")
    fake.course_flow = type("Flow", (), {"serve": staticmethod(fake_serve)})()
    monkeypatch.setitem(sys.modules, "knowledge.pipeline.flows", fake)
    runpy.run_module("knowledge.pipeline.serve", run_name="__main__")

    assert calls and calls[0]["name"] == "course-ingest"
    limit = calls[0]["global_limit"]
    assert limit.limit == 1
    assert limit.collision_strategy is ConcurrencyLimitStrategy.ENQUEUE
