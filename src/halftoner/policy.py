"""Target policies. The math never disagrees between targets; enforcement does.

Checks are always computed. Each target decides what a failing check means:
  report -- listed, render proceeds          warn   -- listed loudly, render proceeds
  cap    -- the target clamps the value      refuse -- render stops unless forced
  ignore -- not relevant to this target
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .ink import InkSet

POLICIES: dict[str, dict[str, str]] = {
    "screen": {"ruling": "report", "min_dot": "report", "sampling": "report", "angle": "report", "geometry": "ignore", "coverage": "report"},
    "svg": {"ruling": "report", "min_dot": "report", "sampling": "report", "angle": "report", "geometry": "warn", "coverage": "report"},
    "pod": {"ruling": "cap", "min_dot": "report", "sampling": "report", "angle": "report", "geometry": "ignore", "coverage": "report"},
    "film": {"ruling": "refuse", "min_dot": "refuse", "sampling": "refuse", "angle": "refuse", "geometry": "ignore", "coverage": "report"},
}
TARGETS = tuple(POLICIES)


class ConstraintRefused(Exception):
    def __init__(self, target: str, checks):
        self.target, self.checks = target, checks
        lines = "\n".join(f"  {c.name}: {c.detail}" for c in checks)
        super().__init__(f"{target} target refuses:\n{lines}\n(force to render anyway)")


@dataclass
class Outcome:
    target: str
    report: object
    actions: list  # (Check, action) for every failing check the target doesn't ignore
    forced: bool = False
    capped: dict[str, tuple[float, float]] = field(default_factory=dict)
    paths: list = field(default_factory=list)

    def __str__(self) -> str:
        out = [f"target {self.target}"]
        for check, action in self.actions:
            if action == "cap" and self.capped:
                continue  # the capped values below say it more precisely
            label = "REFUSED, forced" if action == "refuse" else action
            out.append(f"  [{label}] {check.name}: {check.detail}")
        for ink, (was, now) in self.capped.items():
            out.append(f"  [cap] {ink} ruling {was:g} -> {now:g} lpi")
        out += [f"  wrote {p}" for p in self.paths]
        return "\n".join(out)


def cap_ruling(recipe):
    ceiling = recipe.substrate.ruling_ceiling_lpi
    capped, inks = {}, []
    for ink in recipe.inks:
        lpi = recipe.ruling_for(ink)
        if lpi > ceiling:
            capped[ink.name] = (lpi, ceiling)
            ink = replace(ink, ruling_lpi=ceiling)
        inks.append(ink)
    overprint = {tuple(k): v for k, v in recipe.inks.overprint.items()}
    return replace(recipe, inks=InkSet(*inks, overprint=overprint)), capped


def apply_policy(recipe, target: str, force: bool = False):
    """(recipe to render, Outcome). Raises ConstraintRefused unless forced."""
    if target not in POLICIES:
        raise ValueError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
    policy = POLICIES[target]
    report = recipe.report()
    actions = [(c, policy.get(c.kind, "report")) for c in report.checks if not c.ok]
    actions = [(c, a) for c, a in actions if a != "ignore"]
    refused = [c for c, a in actions if a == "refuse"]
    if refused and not force:
        raise ConstraintRefused(target, refused)
    job, capped = recipe, {}
    if any(a == "cap" and c.kind == "ruling" for c, a in actions):
        job, capped = cap_ruling(recipe)
    return job, Outcome(target, report, actions, forced=bool(refused), capped=capped)
