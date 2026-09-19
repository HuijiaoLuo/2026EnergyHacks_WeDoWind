"""Static relative-heading graph audit for target-specific yaw anchors.

This module is intentionally separate from the release PARS path.  It does
not alter pair/sector baselines, state boundaries, or state amplitudes.  It
only asks whether the *static* pair/sector heading offsets, which are already
removed before state detection, contain a label-free turbine-specific absolute
anchor signal.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from baseline_loto_ridge import wrap_180
from fleet_context import FleetConfig, candidate_neighbours, circular_median_deg
from yaw_farm_anchor import legacy_global_anchor
from yaw_model import predict_from_bundle, score_prediction
from yaw_relative_state import RelativeStateConfig


@dataclass(frozen=True)
class StaticGraphAnchorConfig:
    """Pre-declared settings for the static graph diagnostic.

    Sector medians are given equal influence so the result is not dominated by
    the prevailing wind-direction distribution.  These settings affect only
    the experimental static graph, never the release state detector.
    """

    min_sector_samples: int = 25
    min_sectors: int = 4
    dispersion_scale_deg: float = 8.0
    min_edge_weight: float = 0.01


def _circular_mad_deg(values: np.ndarray, centre: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or not np.isfinite(centre):
        return np.nan
    return float(np.median(np.abs(wrap_180(values - centre))))


def build_static_pair_edges(
    binned: Mapping[str, pd.DataFrame],
    layout: pd.DataFrame,
    fleet_config: FleetConfig,
    relative_config: RelativeStateConfig,
    turbines: Sequence[str] | None = None,
    config: StaticGraphAnchorConfig | None = None,
) -> pd.DataFrame:
    """Estimate frozen sector-baseline offsets for directed neighbour edges.

    For each directed pair ``i -> j``, this recreates the static part of the
    relative-heading preprocessing: ``wrap(NacDir_i - NacDir_j)`` is reduced
    within the neighbour's wind sectors.  The resulting sector medians are
    then combined with an equal-sector circular median.  Dynamic residuals are
    deliberately not used here.
    """
    config = config or StaticGraphAnchorConfig()
    turbine_ids = list(turbines) if turbines is not None else sorted(binned)
    turbine_set = set(turbine_ids)
    n_possible_sectors = int(round(360.0 / relative_config.sector_width_deg))
    rows: list[dict[str, float | int | str]] = []

    for left in turbine_ids:
        if left not in binned or left not in layout.index:
            continue
        candidates = candidate_neighbours(left, layout, fleet_config)
        for right, candidate in candidates.iterrows():
            if right not in turbine_set or right not in binned:
                continue

            left_frame = binned[left][["NacDir"]].copy()
            right_frame = binned[right][["NacDir", "WindDir"]].reindex(
                left_frame.index
            )
            pair = pd.DataFrame(index=left_frame.index)
            pair["angle"] = wrap_180(
                left_frame["NacDir"].to_numpy(dtype=float)
                - right_frame["NacDir"].to_numpy(dtype=float)
            )
            pair["wind"] = right_frame["WindDir"].to_numpy(dtype=float)
            pair = pair.dropna()
            if pair.empty:
                continue

            pair["sector"] = np.floor(
                (pair["wind"].to_numpy(dtype=float) % 360.0)
                / relative_config.sector_width_deg
            ).astype(int)
            sector = pair.groupby("sector", sort=True)["angle"].agg(
                baseline_deg=circular_median_deg,
                n="size",
            )
            sector = sector[sector["n"] >= config.min_sector_samples]
            if len(sector) < config.min_sectors:
                continue

            baselines = sector["baseline_deg"].to_numpy(dtype=float)
            edge_offset = float(circular_median_deg(baselines))
            dispersion = _circular_mad_deg(baselines, edge_offset)
            sector_fraction = float(len(sector) / n_possible_sectors)
            edge_weight = float(
                candidate["distance_weight"]
                * sector_fraction
                * np.exp(-np.square(dispersion / config.dispersion_scale_deg))
            )
            if not np.isfinite(edge_weight) or edge_weight < config.min_edge_weight:
                continue

            rows.append(
                {
                    "left": str(left),
                    "right": str(right),
                    "site": str(layout.loc[left, "site"]),
                    "edge_offset_deg": edge_offset,
                    "edge_sector_mad_deg": dispersion,
                    "n_sector": int(len(sector)),
                    "n_observations": int(sector["n"].sum()),
                    "distance": float(candidate["distance"]),
                    "distance_weight": float(candidate["distance_weight"]),
                    "edge_weight": edge_weight,
                }
            )

    columns = [
        "left",
        "right",
        "site",
        "edge_offset_deg",
        "edge_sector_mad_deg",
        "n_sector",
        "n_observations",
        "distance",
        "distance_weight",
        "edge_weight",
    ]
    return pd.DataFrame(rows, columns=columns)


def _components(edges: pd.DataFrame, nodes: Sequence[str]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {str(node): set() for node in nodes}
    for row in edges.itertuples(index=False):
        adjacency.setdefault(str(row.left), set()).add(str(row.right))
        adjacency.setdefault(str(row.right), set()).add(str(row.left))

    components: list[list[str]] = []
    unseen = set(adjacency)
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        queue: deque[str] = deque([root])
        component = [root]
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
                    component.append(neighbour)
        components.append(sorted(component))
    return components


def fit_static_heading_graph(
    edges: pd.DataFrame,
    nodes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Solve graph coordinates ``g_left - g_right ~= edge_offset``.

    Each connected component is solved independently with one arbitrary node
    set to zero.  Only coordinate differences are used downstream, so this
    gauge choice cannot affect a prediction.
    """
    required = {"left", "right", "edge_offset_deg", "edge_weight"}
    if not required.issubset(edges.columns):
        raise ValueError(f"edges must contain {sorted(required)}")

    usable = edges.copy()
    usable = usable[
        np.isfinite(usable["edge_offset_deg"].to_numpy(dtype=float))
        & np.isfinite(usable["edge_weight"].to_numpy(dtype=float))
        & (usable["edge_weight"].to_numpy(dtype=float) > 0.0)
    ].copy()
    all_nodes = sorted(
        set(map(str, nodes or []))
        | set(usable["left"].astype(str))
        | set(usable["right"].astype(str))
    )
    if not all_nodes:
        return pd.DataFrame(
            columns=[
                "turbine",
                "component",
                "graph_coordinate_deg",
                "incident_weight",
                "n_incident_edges",
            ]
        )

    rows: list[dict[str, float | int | str]] = []
    for component_id, component in enumerate(_components(usable, all_nodes)):
        component_edges = usable[
            usable["left"].astype(str).isin(component)
            & usable["right"].astype(str).isin(component)
        ]
        incident_weight = defaultdict(float)
        incident_count = defaultdict(int)
        for edge in component_edges.itertuples(index=False):
            incident_weight[str(edge.left)] += float(edge.edge_weight)
            incident_weight[str(edge.right)] += float(edge.edge_weight)
            incident_count[str(edge.left)] += 1
            incident_count[str(edge.right)] += 1

        coordinates = {node: np.nan for node in component}
        if len(component) == 1 or component_edges.empty:
            coordinates[component[0]] = 0.0
        else:
            reference = component[0]
            free_nodes = [node for node in component if node != reference]
            index = {node: k for k, node in enumerate(free_nodes)}
            design = np.zeros((len(component_edges), len(free_nodes)), dtype=float)
            target = component_edges["edge_offset_deg"].to_numpy(dtype=float)
            weights = np.sqrt(
                component_edges["edge_weight"].to_numpy(dtype=float)
            )
            for row_id, edge in enumerate(component_edges.itertuples(index=False)):
                if str(edge.left) != reference:
                    design[row_id, index[str(edge.left)]] = 1.0
                if str(edge.right) != reference:
                    design[row_id, index[str(edge.right)]] = -1.0
            solution, *_ = np.linalg.lstsq(
                design * weights[:, None],
                target * weights,
                rcond=None,
            )
            coordinates[reference] = 0.0
            coordinates.update(
                {node: float(solution[k]) for node, k in index.items()}
            )

        for node in component:
            rows.append(
                {
                    "turbine": node,
                    "component": int(component_id),
                    "graph_coordinate_deg": coordinates[node],
                    "incident_weight": float(incident_weight[node]),
                    "n_incident_edges": int(incident_count[node]),
                }
            )
    return pd.DataFrame(rows).sort_values("turbine").reset_index(drop=True)


def graph_anchor_adjustment(
    turbine: str,
    fit_ids: Sequence[str],
    graph: pd.DataFrame,
    direction: float = 1.0,
) -> tuple[float, bool]:
    """Return a target's static graph offset relative to labelled fit nodes.

    A target outside every labelled component is not adjusted.  This is an
    explicit identifiability guard for cross-site targets with no labelled
    connection.
    """
    table = graph.set_index("turbine")
    if turbine not in table.index:
        return 0.0, False
    target = table.loc[turbine]
    if not np.isfinite(target["graph_coordinate_deg"]):
        return 0.0, False

    references = table.reindex(list(fit_ids))
    references = references[
        (references["component"] == target["component"])
        & np.isfinite(references["graph_coordinate_deg"])
    ]
    if references.empty:
        return 0.0, False

    reference_coordinate = float(references["graph_coordinate_deg"].mean())
    raw_delta = float(target["graph_coordinate_deg"] - reference_coordinate)
    return float(direction * raw_delta), True


def run_static_graph_anchor_loto(
    train_ids: Sequence[str],
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
    graph: pd.DataFrame,
    beta: float = 1.0,
) -> pd.DataFrame:
    """Audit sign-symmetric static graph anchor transfer under strict LOTO.

    The release anchor is always included.  ``graph_plus`` and ``graph_minus``
    use the raw graph-coordinate difference in both physically possible signs.
    They are diagnostic alternatives, not an automatically selected model.
    """
    rows: list[dict[str, float | str | bool]] = []
    variants = {"release": 0.0, "graph_plus": 1.0, "graph_minus": -1.0}

    for holdout in train_ids:
        fit_ids = [turbine for turbine in train_ids if turbine != holdout]
        centre = legacy_global_anchor(fit_ids, labels, theta_star)
        for name, direction in variants.items():
            delta, connected = (
                (0.0, True)
                if direction == 0.0
                else graph_anchor_adjustment(holdout, fit_ids, graph, direction)
            )
            prediction = predict_from_bundle(
                holdout,
                stage_bundles[holdout],
                centre + delta,
                beta,
                theta_star,
            )
            metrics = score_prediction(
                prediction,
                labels[holdout],
                centre + delta - theta_star[holdout],
            )
            rows.append(
                {
                    "anchor": name,
                    "holdout": holdout,
                    "C_release": centre,
                    "graph_delta_deg": delta,
                    "graph_connected": bool(connected),
                    "B0": centre + delta - theta_star[holdout],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "constant_mae": metrics["constant_mae"],
                    "constant_rmse": metrics["constant_rmse"],
                    "states": metrics["n_states"],
                }
            )
    return pd.DataFrame(rows)
