import ast
from importlib.util import find_spec
from pathlib import Path


TARGET_MOD = (
    "core.schema.ingest",
    "core.ingest.segment",
    "core.ingest.novelty",
    "knowledge.ingest.fetch",
    "knowledge.ingest.parse",
    "knowledge.ingest.publish",
    "knowledge.ingest.anchor",
    "knowledge.ingest.persist",
    "knowledge.ingest.vid",
)

OLD_MOD = (
    "core.ingest.contracts",
    "core.ingest.stages.fetch",
    "core.ingest.stages.parse",
    "core.ingest.stages.publish",
    "core.ingest.stages.segment",
    "core.ingest.stages.anchor",
    "core.ingest.stages.persist",
    "core.ingest.stages.video",
)

APP_ROOT = Path(__file__).resolve().parents[3]


def _mod_exist(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def test_ingest_mod_follow_r3():
    missing = [name for name in TARGET_MOD if not _mod_exist(name)]
    stale = [name for name in OLD_MOD if _mod_exist(name)]

    assert not missing, f"Missing R3 modules: {missing}"
    assert not stale, f"Stale ingestion modules: {stale}"


def test_prefect_import_stay_pipeline():
    leaks = []
    for path in APP_ROOT.rglob("*.py"):
        rel = path.relative_to(APP_ROOT).as_posix()
        if rel.startswith(("knowledge/pipeline/", "test/")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            mods = ([node.module] if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names] if isinstance(node, ast.Import)
                    else [])
            if any(mod and (mod == "prefect" or mod.startswith("prefect.")) for mod in mods):
                leaks.append(rel)
                break

    assert not leaks, f"Prefect import outside knowledge/pipeline: {leaks}"
