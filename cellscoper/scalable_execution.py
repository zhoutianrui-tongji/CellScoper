"""Resolve an explicit, provenance-friendly execution plan by realized scale.

The profile decides how large outputs are handled: ``exact`` and ``sampled`` keep full diagnostic tables; ``streaming``
(above ``execution.sampled_max_metaspots`` MetaSpots, e.g. full Atera with ~32M) skips them and writes the final
transcript table in cell-aligned parts when ``io.write_partitioned_dataset`` is on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: int
    requested_profile: str
    effective_profile: str
    n_metaspots: int
    n_transcripts: int
    exact_max_metaspots: int
    sampled_max_metaspots: int
    partition_rows: int
    partitioned_annotation: bool
    full_diagnostics: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_PROFILE_ALIASES = {
    "auto": "auto",
    "exact": "exact",
    "full_batch": "exact",
    "full-batch": "exact",
    "sampled": "sampled",
    "streaming": "streaming",
    "large": "streaming",
}


def resolve_execution_plan(
    config: Mapping[str, Any],
    *,
    n_metaspots: int,
    n_transcripts: int = 0,
) -> ExecutionPlan:
    """Resolve the exact / sampled / streaming profile from the realized MetaSpot count."""
    if n_metaspots < 1:
        raise ValueError("Execution planning requires at least one MetaSpot")
    options = dict(config or {}).get("execution", {}) or {}
    if not isinstance(options, Mapping):
        raise ValueError("execution must be a mapping")
    requested_raw = str(options.get("profile", "auto")).strip().lower()
    if requested_raw not in _PROFILE_ALIASES:
        raise ValueError("execution.profile must be one of {'auto','exact','sampled','streaming'}")
    requested = _PROFILE_ALIASES[requested_raw]
    exact_max = int(options.get("exact_max_metaspots", 200_000))
    sampled_max = int(options.get("sampled_max_metaspots", 2_000_000))
    partition_rows = int(options.get("partition_rows", 1_000_000))
    if exact_max < 1 or sampled_max < exact_max or partition_rows < 1:
        raise ValueError(
            "execution thresholds must satisfy 1 <= exact_max_metaspots <= "
            "sampled_max_metaspots and partition_rows >= 1"
        )
    if requested == "auto":
        if n_metaspots <= exact_max:
            effective = "exact"
        elif n_metaspots <= sampled_max:
            effective = "sampled"
        else:
            effective = "streaming"
    else:
        effective = requested
    if effective == "exact" and n_metaspots > exact_max:
        raise ValueError(
            f"execution.profile='exact' received {n_metaspots:,} MetaSpots, "
            f"above execution.exact_max_metaspots={exact_max:,}"
        )
    return ExecutionPlan(
        schema_version=2,
        requested_profile=requested,
        effective_profile=effective,
        n_metaspots=int(n_metaspots),
        n_transcripts=int(n_transcripts),
        exact_max_metaspots=exact_max,
        sampled_max_metaspots=sampled_max,
        partition_rows=partition_rows,
        partitioned_annotation=effective == "streaming",
        full_diagnostics=effective != "streaming",
    )
