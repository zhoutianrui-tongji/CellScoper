import numpy as np
import pandas as pd

from cellscope.evidence_annotation import (
    marker_sets_from_reference,
    resolution_check,
    run_evidence_annotation,
)


FAST = {
    "evidence_null_sets": 8,
    "evidence_bootstrap": 300,
    "evidence_min_gene_transcripts": 20,
}


def _synthetic(seed=0, n_cells=80, shuffle_genes_within_cell=False):
    """Cells with a planted ER patch; OMM markers carry no spatial structure."""
    rng = np.random.default_rng(seed)
    er_markers = [f"ER{i}" for i in range(24)]
    er_partners = [f"SEC{i}" for i in range(24)]       # co-localize, not in reference
    omm_markers = [f"OMM{i}" for i in range(24)]
    nuclear_genes = [f"NUC{i}" for i in range(30)]
    other = [f"G{i}" for i in range(120)]
    rows = []
    for cell in range(n_cells):
        for spot in range(30):
            nuclear = spot < 12
            planted = (not nuclear) and spot >= 24      # 6 ER spots per cell
            if nuclear:
                pool, weights = nuclear_genes + other, [4.0] * 30 + [1.0] * 120
            elif planted:
                pool = er_markers + er_partners + omm_markers + other
                weights = [6.0] * 48 + [1.0] * 24 + [1.0] * 120
            else:
                pool = er_markers + er_partners + omm_markers + other
                weights = [0.3] * 48 + [1.0] * 24 + [1.0] * 120
            p = np.asarray(weights) / np.sum(weights)
            for gene in rng.choice(pool, size=25, p=p):
                rows.append((f"c{cell}", gene, f"c{cell}:{spot}", str(spot % 5), nuclear, planted))
    frame = pd.DataFrame(
        rows,
        columns=["cell_id", "feature_name", "metaspot", "meta-domain", "in_nucleus", "planted"],
    )
    if shuffle_genes_within_cell:
        frame["feature_name"] = frame.groupby("cell_id")["feature_name"].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
    columns = ["Nucleus", "Nucleolus", "Lamina", "Nuclear_Pore", "Cytosol", "ER", "OMM"]
    reference = pd.DataFrame(0.0, index=er_markers + omm_markers, columns=columns)
    reference.loc[er_markers, "ER"] = 1.0
    reference.loc[omm_markers, "OMM"] = 1.0
    return frame, reference


def test_marker_sets_require_a_clear_top_compartment():
    reference = pd.DataFrame(
        {"Nucleus": [0.0, 0.9, 0.0], "ER": [1.0, 1.0, 0.0], "OMM": [0.1, 0.0, 0.0]},
        index=["a", "b", "c"],
    )
    sets = marker_sets_from_reference(reference, min_margin=0.2)
    assert sets["ER"] == ["A"]           # b is ambiguous, c has no evidence
    assert sets["OMM"] == []


def test_planted_structure_is_supported_and_unstructured_markers_are_not():
    frame, reference = _synthetic()
    table, report = run_evidence_annotation(frame, reference, FAST)

    assert report["resolution"]["passed"]
    assert report["structures"]["ER"]["supported"]
    assert not report["structures"]["OMM"]["supported"]
    assert report["supported_structures"] == ["ER"]

    truth = frame.groupby("metaspot")["planted"].first().reindex(table["metaspot"]).to_numpy()
    called = table["subcellular_annotation"].to_numpy() == "ER"
    assert called.sum() > 0
    assert truth[called].mean() > 0.9                    # precision
    assert called[truth].mean() > 0.5                    # recall
    # Nuclear units never receive cytoplasmic structures.
    nuclear = table["subcellular_broad_compartment"].to_numpy() == "Nucleus"
    assert set(table.loc[nuclear, "subcellular_annotation"]) == {"Nucleus"}
    assert "OMM" not in set(table["subcellular_annotation"])


def test_propagation_uses_genes_outside_the_reference():
    frame, reference = _synthetic()
    few = reference.loc[[f"ER{i}" for i in range(6)] + [f"OMM{i}" for i in range(24)]]
    truth = frame.groupby("metaspot")["planted"].first()

    def recall(mode):
        table, _ = run_evidence_annotation(
            frame, few, {**FAST, "evidence_score": mode, "evidence_unit": "metaspot"}
        )
        called = table.set_index("metaspot")["subcellular_annotation_exploratory"] == "ER"
        return float(called[truth[called.index].to_numpy()].mean())

    assert recall("propagated") > recall("marker") + 0.2


def test_resolution_gate_reports_broad_labels_when_positions_carry_no_signal():
    frame, reference = _synthetic(shuffle_genes_within_cell=True)
    check = resolution_check(frame["cell_id"], frame["feature_name"], frame["in_nucleus"])
    assert check["split_half_spearman"] < 0.3

    table, report = run_evidence_annotation(frame, reference, FAST)
    assert not report["resolution"]["passed"]
    assert report["supported_structures"] == []
    assert set(table["subcellular_annotation"]) == {"Nucleus", "Cytosol"}
