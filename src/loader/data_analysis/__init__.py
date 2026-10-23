"""Meddef-style dataset audits adapted for Viola (DD / QuickDraw / SketchyScene / TU-Berlin)."""

__all__ = ["run_full_audit"]


def __getattr__(name: str):
    if name == "run_full_audit":
        from .check import run_full_audit

        return run_full_audit
    raise AttributeError(name)
