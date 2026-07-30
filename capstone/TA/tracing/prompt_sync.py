"""
TA/tracing/prompt_sync.py

Prompt synchronization to Langfuse — part of the tracing infrastructure.

WHY this is here (not in workflow/sync_prompts.py):
  - Uses langfuse_config centrally (no dotenv hacks)
  - Reliable variable escaping that handles all edge cases
  - Post-push validation to guarantee Langfuse received the correct template
  - Dry-run support for verification before touching production
  - Can be imported and called programmatically from anywhere

ESCAPING STRATEGY:
  Python local format uses  : {var}
  Langfuse template format  : {{var}}

  The naive regex r'\\{([a-zA-Z0-9_]+)\\}' fails when:
    1. Template already contains {{ or }} (dict literals, JSON examples, etc.)
    2. Nested braces in few-shot annotation text
    3. Prompt contains Python format *examples* inside strings

  Safe approach:
    1. Protect existing {{ and }} with sentinels
    2. Convert remaining single {var} → {{var}}
    3. Restore sentinels

VALIDATION STRATEGY:
  After push, fetch the prompt back from Langfuse and compare
  the resolved variable set against expected variables extracted
  from the original Python template.
"""

import re
import logging
from typing import Dict, Tuple

from core.config import langfuse_config

logger = logging.getLogger(__name__)

# ─── Sentinel values (must never appear in prompt text) ───────────────────────
_OPEN_SENTINEL  = "\x00LFOPEN\x00"
_CLOSE_SENTINEL = "\x00LFCLOSE\x00"


def _python_to_langfuse(template: str) -> Tuple[str, list[str]]:
    """
    Convert a Python format-string template to Langfuse {{var}} format.

    Returns:
        lf_template : the escaped string ready for Langfuse
        variables   : list of variable names found (for validation)

    Steps:
        1. Protect existing {{ and }} with sentinels
        2. Extract and convert all {var} → {{var}}
        3. Restore sentinels → {{ and }}
    """
    # Step 1: protect already-doubled braces
    protected = template.replace("{{", _OPEN_SENTINEL).replace("}}", _CLOSE_SENTINEL)

    # Step 2: find all single {var} and collect variable names
    found_vars = re.findall(r'\{([a-zA-Z0-9_]+)\}', protected)
    converted  = re.sub(r'\{([a-zA-Z0-9_]+)\}', r'{{\1}}', protected)

    # Step 3: restore sentinels
    lf_template = converted.replace(_OPEN_SENTINEL, "{{").replace(_CLOSE_SENTINEL, "}}")

    return lf_template, list(dict.fromkeys(found_vars))  # deduplicated, order-preserved


def _extract_lf_variables(lf_template: str) -> list[str]:
    """Extract variable names from a Langfuse {{var}} template."""
    return list(dict.fromkeys(re.findall(r'\{\{([a-zA-Z0-9_]+)\}\}', lf_template)))


class PromptSyncer:
    """
    Syncs prompt templates from the local codebase to Langfuse.

    Usage:
        syncer = PromptSyncer()
        syncer.sync_all(dry_run=True)   # validate only
        syncer.sync_all()               # push to Langfuse production
    """

    def __init__(self):
        if not langfuse_config.enable:
            logger.warning("[PromptSyncer] langfuse_config.enable=False. Sync is a no-op.")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                secret_key=langfuse_config.key,
                public_key=langfuse_config.public_key,
                host=langfuse_config.host,
            )
            return self._client
        except Exception as e:
            raise RuntimeError(f"[PromptSyncer] Cannot connect to Langfuse: {e}") from e

    def _collect_prompts(self) -> Dict[str, str]:
        """Import and flatten all prompt templates from prompt.py."""
        from TA.helper.prompt import (
            _ROUTER_PROMPT,
            _RETRIEVE_REFINE_PROMPT,
            _DEEP_CHECK_PROMPT,
            _RESEARCH_STRATEGY_PROMPT,
            _ROADMAP_PROMPT,
            _TEACH_UNDERSTAND_PROMPT,
            _TEACH_REVIEW_PROMPT,
            _TEACH_CONTINUE_PROMPT,
            _TEACH_EVAL_PROMPT_V2,
            _NEXT_TOPIC_PROMPT,
            _PROPOSAL_PRESENT_PROMPT,
            _UNKNOWN_PROMPT,
        )

        prompts: Dict[str, str] = {
            "TA_ROUTER_PROMPT":           _ROUTER_PROMPT,
            "TA_RETRIEVE_REFINE_PROMPT":  _RETRIEVE_REFINE_PROMPT,
            "TA_DEEP_CHECK_PROMPT":       _DEEP_CHECK_PROMPT,
            "TA_TEACH_UNDERSTAND_PROMPT": _TEACH_UNDERSTAND_PROMPT,
            "TA_TEACH_REVIEW_PROMPT":     _TEACH_REVIEW_PROMPT,
            "TA_TEACH_CONTINUE_PROMPT":   _TEACH_CONTINUE_PROMPT,
            "TA_TEACH_EVAL_PROMPT_V2":    _TEACH_EVAL_PROMPT_V2,
            "TA_NEXT_TOPIC_PROMPT":       _NEXT_TOPIC_PROMPT,
            "TA_PROPOSAL_PRESENT_PROMPT": _PROPOSAL_PRESENT_PROMPT,
            "TA_UNKNOWN_PROMPT":          _UNKNOWN_PROMPT,
        }

        for k, v in _RESEARCH_STRATEGY_PROMPT.items():
            prompts[f"TA_RESEARCH_STRATEGY_PROMPT_{k.upper()}"] = v
        for k, v in _ROADMAP_PROMPT.items():
            prompts[f"TA_ROADMAP_PROMPT_{k.upper()}"] = v

        return prompts

    def validate(self) -> bool:
        """
        Dry-run: convert all templates and report detected variables.
        Does NOT push to Langfuse. Returns True if all prompts converted cleanly.
        """
        prompts = self._collect_prompts()
        ok = True
        print(f"\n{'─'*60}")
        print(f"[PromptSyncer] DRY RUN — {len(prompts)} prompts")
        print(f"{'─'*60}")
        for name, raw in prompts.items():
            lf_template, variables = _python_to_langfuse(raw)
            round_trip_vars = _extract_lf_variables(lf_template)
            match = sorted(variables) == sorted(round_trip_vars)
            status = "✅" if match else "❌ MISMATCH"
            print(f"  {status}  {name}")
            print(f"         vars: {variables}")
            if not match:
                print(f"         lf_vars (mismatch): {round_trip_vars}")
                ok = False
        print(f"{'─'*60}")
        print(f"Result: {'ALL OK' if ok else 'ISSUES FOUND — fix before pushing'}\n")
        return ok

    def sync_all(self, dry_run: bool = False, labels: list[str] = None) -> None:
        """
        Push all prompts to Langfuse with validation.

        Args:
            dry_run: If True, only validates templates without pushing.
            labels:  Langfuse labels. Defaults to ["production"].
        """
        if dry_run:
            self.validate()
            return

        if not langfuse_config.enable:
            print("[PromptSyncer] Langfuse disabled in config. Nothing pushed.")
            return

        labels = labels or ["production"]
        client = self._get_client()
        prompts = self._collect_prompts()

        print(f"\n{'─'*60}")
        print(f"[PromptSyncer] Syncing {len(prompts)} prompts → {langfuse_config.host}")
        print(f"{'─'*60}")

        success, failed = 0, []
        for name, raw in prompts.items():
            lf_template, variables = _python_to_langfuse(raw)
            try:
                client.create_prompt(
                    name=name,
                    type="text",
                    prompt=lf_template,
                    labels=labels,
                    config={"variables": variables},   # explicit variable metadata
                )

                # ── Post-push validation ─────────────────────────────
                fetched = client.get_prompt(name, type="text")
                fetched_content = fetched.get_langchain_prompt()
                fetched_vars = re.findall(r'\{([a-zA-Z0-9_]+)\}', fetched_content)

                missing = set(variables) - set(fetched_vars)
                extra   = set(fetched_vars) - set(variables)

                if missing or extra:
                    print(f"  ⚠️  {name}: validation mismatch")
                    if missing: print(f"       missing in Langfuse: {missing}")
                    if extra:   print(f"       extra in Langfuse:   {extra}")
                    failed.append(name)
                else:
                    print(f"  ✅  {name}  ({len(variables)} vars: {variables})")
                    success += 1

            except Exception as e:
                print(f"  ❌  {name}: {e}")
                failed.append(name)

        client.flush()
        print(f"{'─'*60}")
        print(f"Done: {success} pushed, {len(failed)} failed")
        if failed:
            print(f"Failed: {failed}")
        print()


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    dry = "--dry" in sys.argv or "--validate" in sys.argv
    syncer = PromptSyncer()
    syncer.sync_all(dry_run=dry)
