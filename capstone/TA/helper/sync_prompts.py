"""
DEPRECATED — logic đang ở TA/tracing/prompt_sync.py
Wrapper để ko bị lỗi import

Ko vi phạm Dependency rules (dù ko clean cho lắm)
"""

from TA.tracing.prompt_sync import PromptSyncer as _Syncer


def sync_all_prompts():
    """Deprecated wrapper. Use PromptSyncer().sync_all() directly."""
    _Syncer().sync_all()


if __name__ == "__main__":
    import sys
    dry = "--dry" in sys.argv or "--validate" in sys.argv
    _Syncer().sync_all(dry_run=dry)
