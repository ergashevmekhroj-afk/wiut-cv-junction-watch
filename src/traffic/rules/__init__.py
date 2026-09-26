"""Event rules. Each rule maps a Context to a list of (start, end) intervals."""
from __future__ import annotations

from .context import Context
from .interaction_rules import accident, near_miss
from .motion_rules import congestion, illegal_u_turn, stopped_vehicle, wrong_way
from .people_rules import failure_to_yield, jaywalking
from .signal_rules import red_light, stop_line

RULES = {
    "accident": accident,
    "near_miss": near_miss,
    "red_light": red_light,
    "stop_line": stop_line,
    "wrong_way": wrong_way,
    "illegal_u_turn": illegal_u_turn,
    "stopped_vehicle": stopped_vehicle,
    "jaywalking": jaywalking,
    "failure_to_yield": failure_to_yield,
    "congestion": congestion,
}


def detect_all(ctx: Context, enabled: set[str]) -> dict[str, list[tuple[float, float]]]:
    return {label: fn(ctx) for label, fn in RULES.items() if label in enabled}
