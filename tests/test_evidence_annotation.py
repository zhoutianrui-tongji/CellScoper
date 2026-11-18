import numpy as np
import pandas as pd

from cellscoper.evidence_annotation import (
    marker_sets_from_reference,
    resolution_check,
    run_evidence_annotation,
)


FAST = {"evidence_bootstrap": 300, "evidence_min_gene_transcripts": 20}


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


def test_resolution_gate_reports_broad_labels_when_positions_carry_no_signal():
    frame, reference = _synthetic(shuffle_genes_within_cell=True)
    check = resolution_check(frame["cell_id"], frame["feature_name"], frame["in_nucleus"])
    assert check["split_half_spearman"] < 0.3

    table, report = run_evidence_annotation(frame, reference, FAST)
    assert not report["resolution"]["passed"]
    assert report["supported_structures"] == []
    assert set(table["subcellular_annotation_validated"]) == {"Nucleus", "Cytosol"}


def test_every_structure_gets_one_seat_per_cell_for_display():
    from cellscoper.evidence_annotation import _seat_structures

    rng = np.random.default_rng(0)
    cells = np.repeat(np.arange(5), 20)
    seats = _seat_structures(rng.normal(size=(100, 2)), cells)
    assert (np.bincount(seats[seats >= 0], minlength=2) == 5).all()   # one seat per cell and structure
    for c in range(5):
        assert len(set(seats[cells == c]) - {-1}) == 2


def test_bundled_extra_markers_extend_the_default_reference():
    from cellscoper.evidence_annotation import marker_sets_from_reference
    from cellscoper.reference import _load_reference_scores

    reference = _load_reference_scores(None, {})
    assert reference.index.is_unique
    sets = marker_sets_from_reference(reference, 0.2)
    assert len(sets["Nuclear Speckle"]) > 100
    extra = pd.read_table("src/extra_reference_markers.tsv")
    listed = set(extra.loc[extra["structure"] == "OMM", "gene"].str.upper())  # the only OMM source
    assert listed <= set(sets["OMM"])
