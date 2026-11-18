"""Session guard: never start a run that cannot finish within the Kaggle session.

A run starts only if  elapsed + safety x predicted + end margin <= guard line,
all in hours since the session started.
"""

from __future__ import annotations


def may_start(elapsed_h: float, predicted_h: float, session_cfg: dict) -> bool:
    need = elapsed_h + session_cfg["run_time_safety_factor"] * predicted_h + session_cfg["end_margin_hours"]
    return need <= session_cfg["guard_line_hours"]
