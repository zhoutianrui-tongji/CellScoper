"""Reference localisation tables for annotation: RNALocate v3, APEX-seq and the bundled extra markers, merged into one
gene x structure score table."""

from typing import Any, Mapping, Optional, Tuple, Union
import logging as _logging
import os
import re as _re
import numpy as np
import pandas as pd

from .resource_paths import bundled_resource_path


SEVEN_KEYS   = ['NUCLEUS','NUCLEOLUS','LAMINA','NUCLEAR_PORE','CYTOSOL','ER','OMM']

SEVEN_PRETTY = {'NUCLEUS':'Nucleus','NUCLEOLUS':'Nucleolus','LAMINA':'Lamina',
                'NUCLEAR_PORE':'Nuclear_Pore','CYTOSOL':'Cytosol','ER':'ER','OMM':'OMM'}

SEVEN_COLS   = [SEVEN_PRETTY[k] for k in SEVEN_KEYS]

L1 = [
    'NUCLEUS','NUC_LEM','NUC_PORE','NUCLEOLUS','CYTOSOL','RNP_GRANULE','RIBOSOME_CYTO',
    'ER','GOLGI','ENDOLYSO','PEROXISOME','MITO_OMM','MITO_IMM','MITO_MATRIX','MITO_UNSPEC',
    'PM_JUNCTION','CYTO_CORTEX_POLARITY','NEURONAL','CELL_DIVISION','EV','RIBOSOME_ORGANELLAR','OTHER',
]

_MULTI_SEP = _re.compile(r'[;,/|]+')

def _norm_loc_txt(s: str) -> str:
    return _re.sub(r'\s+', ' ', str(s).strip().lower())

# Explicit mapping (subset from user-provided list; extended as needed)
L1_EXPLICIT = {
    'nucleolus': 'NUCLEOLUS',
    'nucleus(exclusion from nucleoli)': 'NUCLEUS', 'nucleus': 'NUCLEUS', 'nuclear ': 'NUCLEUS', 'nucleoplasm': 'NUCLEUS',
    'nuclear envelope': 'NUC_LEM', 'nuclear inner membrane': 'NUC_LEM', 'nuclear outer membrane': 'NUC_LEM', 'nuclear periphery': 'NUC_LEM',
    'nuclear lamina': 'NUC_LEM', 'nuclear membrane': 'NUC_LEM',
    'nuclear pore complex': 'NUC_PORE', 'nuclear pore': 'NUC_PORE', 'npc': 'NUC_PORE',
    'cytosol': 'CYTOSOL', 'cytoplasm': 'CYTOSOL', 'cytosolic': 'CYTOSOL', 'perinuclear ': 'CYTOSOL',
    'stress granule': 'RNP_GRANULE', 'p-body': 'RNP_GRANULE',
    'cytosolic ribosome': 'RIBOSOME_CYTO', 'ribosome': 'RIBOSOME_CYTO',
    'endoplasmic reticulum': 'ER', 'er lumen': 'ER', 'er membrane': 'ER', 'rough endoplasmic reticulum': 'ER',
    'golgi apparatus': 'GOLGI', 'endosome': 'ENDOLYSO', 'lysosome': 'ENDOLYSO', 'autophagosome': 'ENDOLYSO',
    'peroxisome': 'PEROXISOME',
    'mitochondrial outer membrane': 'MITO_OMM', 'outer mitochondrial membrane': 'MITO_OMM',
    'mitochondrial inner membrane': 'MITO_IMM', 'mitochondrial matrix': 'MITO_MATRIX',
    'mitochondrial ribosome': 'MITO_UNSPEC', 'mitochondrion': 'MITO_UNSPEC',
    'plasma membrane': 'PM_JUNCTION', 'tight junction': 'PM_JUNCTION', 'cell junction': 'PM_JUNCTION',
    'cell cortex': 'CYTO_CORTEX_POLARITY', 'lamellipodium': 'CYTO_CORTEX_POLARITY', 'pseudopodium': 'CYTO_CORTEX_POLARITY',
    'axon': 'NEURONAL', 'dendrite': 'NEURONAL', 'synapse': 'NEURONAL',
    'centrosome': 'CELL_DIVISION', 'spindle': 'CELL_DIVISION',
    'extracellular vesicle': 'EV', 'extracellular exosome': 'EV',
    'vesicle': 'OTHER', 'membrane': 'OTHER',
}

L1_REGEX = [
    ('NUCLEOLUS',     r'\bnucleol\w+'),
    ('NUC_LEM',       r'\bnuclear (envelope|lamina)\b|\binner nuclear membrane\b|\bouter nuclear membrane\b|\bnuclear periphery\b'),
    ('NUC_PORE',      r'\bnuclear pore( complex)?\b|\bNPC\b'),
    ('NUCLEUS',       r'\bnucleus\b|\bnuclear(?! (envelope|lamina|pore))\b|\bchromatin\b'),
    ('CYTOSOL',       r'\bcytosol(ic)?\b|\bcytoplasm(ic)?\b|\bperinuclear(?! (envelope|membrane))\b'),
    ('RNP_GRANULE',   r'\bstress granule\b|\bP-?body\b|\bRNP\b'),
    ('ER',            r'\b(endoplasmic reticulum|ER)\b|\ber lumen\b|\ber membrane\b'),
    ('MITO_OMM',      r'\bouter (mitochondrial|mitochondrion) membrane\b|\bmitochondrial outer membrane\b'),
    ('MITO_IMM',      r'\bmitochondrial inner membrane\b'),
    ('MITO_UNSPEC',   r'\bmitochondri(on|a)l? (?!inner|outer)\b|\bmitochondrion\b'),
    ('PM_JUNCTION',   r'\bplasma membrane\b|\b(cell )?junction\b|\btight junction\b'),
    ('CYTO_CORTEX_POLARITY', r'\bcortex\b|\bapical\b|\bbasal\b|\blamellipod\w+|\bpseudopod\w+|\bprotrusion\b'),
    ('NEURONAL',      r'\baxon\w*|\bdendrit\w*|\bsynaps\w*'),
    ('ENDOLYSO',      r'\bendosome\b|\blysosome\b|\bautophagosome\b'),
    ('EV',            r'\bextracellular (vesicle|exosome)\b|\b(apoptotic|micro|nano)vesicle\b'),
    ('CELL_DIVISION', r'\bcentrosome\b|\bspindle\b|\bmitotic apparatus\b'),
    ('PEROXISOME',    r'\bperoxisome\b'),
    ('GOLGI',         r'\bgolgi\b'),
]

def map_loc_to_l1(loc: str) -> str:
    if not isinstance(loc, str) or not loc.strip():
        return 'OTHER'
    s = _norm_loc_txt(loc)
    for key in sorted(L1_EXPLICIT.keys(), key=len, reverse=True):
        if key and key in s:
            return L1_EXPLICIT[key]
    for l1, patt in L1_REGEX:
        if _re.search(patt, s, flags=_re.IGNORECASE):
            return l1
    return 'OTHER'

def build_gold_from_rnalocate_l1(rna_locate_df: pd.DataFrame,
                                 species=("Homo sapiens","Mus musculus"),
                                 rna_types=("mRNA",),
                                 score_col='RNALocate_Score',
                                 agg_mode='prob_or',
                                 score_floor=0.5,
                                 use_pubmed_weight=True) -> pd.DataFrame:
    need = {'Species','RNA_Symbol','RNA_Type','Subcellular_Localization',score_col}
    miss = need - set(rna_locate_df.columns)
    if miss:
        raise ValueError(f"RNALocate is missing required columns: {miss}")
    df = rna_locate_df.copy()
    if species is not None:
        df = df[df['Species'].isin(species)]
    if rna_types is not None:
        df = df[df['RNA_Type'].isin(rna_types)]
    def _clean_symbol(x):
        if not isinstance(x, str): return None
        x = x.strip(); return x.upper() if x else None
    df['Common_Gene'] = df['RNA_Symbol'].map(_clean_symbol)
    df = df[df['Common_Gene'].notna()]
    df = df.assign(Subcellular_Localization=df['Subcellular_Localization'].astype(str))
    df = df.assign(_loc_split=df['Subcellular_Localization'].map(lambda s: [t.strip() for t in _MULTI_SEP.split(s) if t.strip()]))
    df = df.explode('_loc_split', ignore_index=True)
    df['L1'] = df['_loc_split'].map(map_loc_to_l1)
    df['score'] = pd.to_numeric(df[score_col], errors='coerce').fillna(0.0).clip(0,1)
    df = df[df['score'] >= float(score_floor)]
    if use_pubmed_weight and 'PubMed_ID' in df.columns:
        pm = df['PubMed_ID'].notna() & (df['PubMed_ID'].astype(str).str.strip()!='')
        df.loc[pm,'score'] = (df.loc[pm,'score'] + 0.05).clip(0,1)
    if agg_mode == 'prob_or':
        agg = df.groupby(['Common_Gene','L1'])['score'].apply(lambda s: 1.0 - np.prod(1.0 - s.values))
    elif agg_mode == 'max':
        agg = df.groupby(['Common_Gene','L1'])['score'].max()
    else:
        raise ValueError("'agg_mode' supports only 'prob_or' or 'max'.")
    tab = agg.unstack('L1').reindex(columns=L1).fillna(0.0)
    tab.index.name = 'Common_Gene'
    return tab

def project_l1_to_seven(l1_df: pd.DataFrame, mode='strict', mito_non_omm=None, fold_misc_to_cytosol=True) -> pd.DataFrame:
    def _get(col):
        return pd.to_numeric(l1_df.get(col, pd.Series(0.0, index=l1_df.index)), errors='coerce').fillna(0.0).astype(float)
    out = pd.DataFrame(index=l1_df.index, columns=SEVEN_COLS, dtype=float)
    out['Nucleus']       = _get('NUCLEUS')
    out['Nucleolus']     = _get('NUCLEOLUS')
    out['Lamina']        = _get('NUC_LEM')
    out['Nuclear_Pore']  = _get('NUC_PORE')
    cyt_base = _get('CYTOSOL') + _get('RNP_GRANULE') + _get('RIBOSOME_CYTO')
    if mode == 'relaxed' and fold_misc_to_cytosol:
        cyt_base = (cyt_base + _get('GOLGI') + _get('ENDOLYSO') + _get('PEROXISOME') + _get('PM_JUNCTION') + _get('CYTO_CORTEX_POLARITY') + _get('CELL_DIVISION'))
    out['Cytosol'] = np.clip(cyt_base, 0.0, 1.0)
    out['ER'] = _get('ER')
    omm = _get('MITO_OMM')
    if mode == 'relaxed' and mito_non_omm in ('OMM','Cytosol'):
        add_mito = _get('MITO_IMM') + _get('MITO_MATRIX') + _get('MITO_UNSPEC')
        if mito_non_omm == 'OMM':
            omm = 1.0 - (1.0 - omm) * (1.0 - add_mito)
        elif mito_non_omm == 'Cytosol':
            out['Cytosol'] = 1.0 - (1.0 - out['Cytosol']) * (1.0 - add_mito)
    out['OMM'] = np.clip(omm, 0.0, 1.0)
    out = out.fillna(0.0); out.index.name = 'Common_Gene'
    return out

def _dedup_columns_by(df: pd.DataFrame, how='prob_or'):
    if not df.columns.duplicated().any():
        return df
    X = df.T
    if how == 'prob_or':
        def por(x):
            x = pd.to_numeric(x, errors='coerce').fillna(0.0).astype(float)
            return 1.0 - np.prod(1.0 - x.values)
        agg = X.groupby(level=0, sort=False).agg(por)
    elif how == 'max':
        agg = X.groupby(level=0, sort=False).max()
    else:
        raise ValueError("'how' supports only 'prob_or' or 'max'.")
    return agg.T

def _dedup_index_by(df: pd.DataFrame, how='prob_or'):
    idx_up = df.index.astype(str).str.upper()
    X = df.copy(); X.index = idx_up
    if not idx_up.duplicated().any():
        return X
    if how == 'prob_or':
        def por(x):
            x = pd.to_numeric(x, errors='coerce').fillna(0.0).astype(float)
            return 1.0 - np.prod(1.0 - x.values)
        return X.groupby(level=0, sort=False).agg(por)
    elif how == 'max':
        return X.groupby(level=0, sort=False).max()
    else:
        raise ValueError("'how' supports only 'prob_or' or 'max'.")

def merge_gold(existing_gold_df: pd.DataFrame, new_gold_df: pd.DataFrame, fuse='max'):
    col_alias = {
        'NUCLEUS':'Nucleus','NUCLEOLUS':'Nucleolus','LAMINA':'Lamina',
        'NUCLEAR_PORE':'Nuclear_Pore','CYTOSOL':'Cytosol','ER':'ER','OMM':'OMM',
        'ERM':'ER','ER_LUMEN':'ER','ER_Lumen':'ER','ER Membrane':'ER','ER membrane':'ER'
    }
    agg_how = 'prob_or' if fuse=='prob_or' else 'max'
    def _prep(df):
        X = df.copy()
        if 'Common_Gene' in X.columns and X.index.name != 'Common_Gene':
            X = X.set_index('Common_Gene')
        X.columns = pd.Index([str(c).strip() for c in X.columns])
        X = X.rename(columns=col_alias)
        if X.columns.duplicated().any():
            X = _dedup_columns_by(X, how=agg_how)
        keep = [c for c in X.columns if c in SEVEN_COLS]
        X = X[keep]
        for c in SEVEN_COLS:
            if c not in X.columns:
                X[c] = 0.0
        X = X.reindex(columns=SEVEN_COLS, copy=False)
        X = _dedup_index_by(X, how=agg_how)
        X = X.apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0.0).astype(float)
        assert X.columns.is_unique, "Column names are still not unique. Please check upstream data or alias mapping."
        return X
    A = _prep(existing_gold_df)
    B = _prep(new_gold_df)
    all_idx = A.index.union(B.index)
    A = A.reindex(all_idx).fillna(0.0)
    B = B.reindex(all_idx).fillna(0.0)
    if fuse == 'prob_or':
        out = 1.0 - (1.0 - A) * (1.0 - B)
    elif fuse == 'max':
        out = np.maximum(A, B)
    else:
        raise ValueError("'fuse' supports only 'max' or 'prob_or'.")
    out.index.name = 'Common_Gene'
    return out

def _prepare_apex_gold_reference(
    apex_raw: pd.DataFrame,
    *,
    omm_specificity_filter: bool = True,
    omm_min_log2fc: float = 0.75,
    omm_min_delta: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the bundled APEX reference and audit OMM specificity.

    APEX-seq enrichment is a soft localization reference, not a census of
    molecules on an organelle.  In particular, the published OMM and ERM
    programs share many secretory-pathway RNAs.  An RNA is therefore allowed
    into the *OMM-specific* reference only when its raw OMM enrichment clears
    both a basal threshold and the stronger of ERM/Cytosol by ``omm_min_delta``.

    The gate is evaluated on the original log2 fold changes.  Row-wise
    normalization is performed afterwards, and failed rows have only their OMM
    value set to zero.  Their ER, Cytosol, and other localization evidence is
    deliberately retained.
    """
    required = {
        "Common_Gene",
        "OMM_log2FC",
        "ERM_log2FC",
        "Cytosol_log2FC",
    }
    missing = sorted(required - set(apex_raw.columns))
    if missing:
        raise ValueError(
            "Bundled APEX reference is missing raw columns required for the "
            f"OMM specificity audit: {missing}"
        )
    try:
        min_log2fc = float(omm_min_log2fc)
        min_delta = float(omm_min_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "APEX OMM thresholds must be finite numeric values"
        ) from exc
    if not np.isfinite(min_log2fc) or not np.isfinite(min_delta):
        raise ValueError("APEX OMM thresholds must be finite numeric values")
    if min_delta < 0.0:
        raise ValueError("apex_omm_min_delta must be non-negative")

    raw = apex_raw.copy()
    omm = pd.to_numeric(raw["OMM_log2FC"], errors="coerce")
    erm = pd.to_numeric(raw["ERM_log2FC"], errors="coerce")
    cytosol = pd.to_numeric(raw["Cytosol_log2FC"], errors="coerce")
    valid = omm.notna() & erm.notna() & cytosol.notna()
    competitor = pd.concat(
        [erm.rename("ERM"), cytosol.rename("Cytosol")],
        axis=1,
    ).max(axis=1)
    delta = omm - competitor
    basal_pass = valid & omm.ge(min_log2fc)
    specific_pass = basal_pass & delta.ge(min_delta)
    shared = basal_pass & ~specific_pass

    reason = pd.Series("invalid_numeric", index=raw.index, dtype="object")
    reason.loc[valid & ~basal_pass] = "below_basal"
    reason.loc[shared] = "shared_with_erm_or_cytosol"
    reason.loc[specific_pass] = "specific"
    audit = pd.DataFrame(
        {
            "Common_Gene": raw["Common_Gene"],
            "Source": raw.get(
                "Source",
                pd.Series("", index=raw.index, dtype="object"),
            ),
            "OMM_log2FC": omm,
            "ERM_log2FC": erm,
            "Cytosol_log2FC": cytosol,
            "omm_competitor_log2fc": competitor,
            "omm_specificity_delta": delta,
            "omm_basal_pass": basal_pass.astype(bool),
            "omm_specific_pass": specific_pass.astype(bool),
            "omm_shared": shared.astype(bool),
            "omm_filter_reason": reason,
        }
    ).reset_index(drop=True)

    cols = ["Common_Gene"] + [
        column
        for column in raw.columns
        if str(column).endswith("_log2FC")
    ]
    gold = raw[cols].rename(
        columns={
            column: column.replace("_log2FC", "")
            for column in cols
            if column != "Common_Gene"
        }
    )
    complete = gold.notna().all(axis=1)
    gold = gold.loc[complete].copy()
    values_columns = [
        column for column in gold.columns if column != "Common_Gene"
    ]
    values = (
        gold[values_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    values = values.subtract(values.min(axis=1), axis=0)
    span = (values.max(axis=1) - values.min(axis=1)).replace(0, 1.0)
    values = values.div(span, axis=0)
    if bool(omm_specificity_filter) and "OMM" in values.columns:
        values.loc[~specific_pass.reindex(values.index).fillna(False), "OMM"] = 0.0
    gold = pd.concat(
        [
            gold[["Common_Gene"]].reset_index(drop=True),
            values.reset_index(drop=True),
        ],
        axis=1,
    )
    return gold, audit

def _load_default_gold_tables(
    *,
    apex_omm_specificity_filter: bool = True,
    apex_omm_min_log2fc: float = 0.75,
    apex_omm_min_delta: float = 0.50,
    return_omm_audit: bool = False,
):
    apex = bundled_resource_path("2019_CELL_APEXSeq.tsv")
    rna = bundled_resource_path("RNALocateV3.txt")
    if not os.path.exists(apex) or not os.path.exists(rna):
        raise FileNotFoundError(
            "Bundled APEX/RNALocate reference files are missing from the "
            "CellScoper installation"
        )
    apex_raw = pd.read_table(apex)
    rna_locate = pd.read_table(rna)
    gold_apex, omm_audit = _prepare_apex_gold_reference(
        apex_raw,
        omm_specificity_filter=apex_omm_specificity_filter,
        omm_min_log2fc=apex_omm_min_log2fc,
        omm_min_delta=apex_omm_min_delta,
    )
    if return_omm_audit:
        return gold_apex, rna_locate, omm_audit
    return gold_apex, rna_locate

def _apply_extra_reference_markers(reference: pd.DataFrame) -> pd.DataFrame:
    """Add the bundled markers that improved held-out validation.

    ``extra_reference_markers.tsv`` lists genes from CAP-seq (ER), APEX-seq
    RNA-dependent OMM mRNAs (OMM) and, for nuclear structures, genes placed by
    genome position: constitutive LADs (Lamina), SON TSA-seq (Nuclear_Speckle)
    and nucleolar-protein TSA-seq (Nucleolus). Nuclear Pore gets no markers. Each listed gene becomes an unambiguous marker
    of its structure; nuclear speckles are a class the APEX-seq/RNALocate
    reference does not have.
    """
    path = bundled_resource_path("extra_reference_markers.tsv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Bundled extra_reference_markers.tsv is missing from the "
            "CellScoper installation"
        )
    extra = pd.read_table(path)
    extra["gene"] = extra["gene"].astype(str)
    reference = reference.copy()
    # OMM markers come only from the bundled list: mRNAs found at the outer
    # mitochondrial membrane by at least two independent methods (APEX-seq, CAP-seq,
    # LOCL-TL, SYNJ2BP RIP) and without a signal peptide. APEX-seq OMM lists alone
    # include ER-translated secretory mRNAs labelled across ER-mitochondria contacts.
    # Nuclear structures are defined by where a gene sits in the genome, which
    # holds in every cell type: Lamina = TSS in a constitutive LMNB1 LAD,
    # Nucleolus = top nucleolar-protein TSA-seq, Nuclear_Speckle = top SON TSA-seq. On Atera
    # BRCA 5k the cLAD genes' nuclear transcripts sit nearer the nuclear rim and
    # the z top/bottom (beyond nuclear fraction and expression) and each set
    # marks its own region far better than the APEX-seq/Halo-seq mRNA lists.
    for column in ("OMM", "Lamina", "Nucleolus"):
        if column in reference.columns and (extra["structure"] == column).any():
            reference[column] = 0.0
    # Nuclear Pore has no genome-position set; its ~20 APEX-seq/RNALocate genes never
    # validated and, as a noisy competitor, decided other structures' specificity.
    # Without markers it is reported as having none and is not placed.
    if "Nuclear_Pore" in reference.columns:
        reference["Nuclear_Pore"] = 0.0
    for column in extra["structure"].unique():
        if column not in reference.columns:
            reference[column] = 0.0
    known_upper = pd.Series(reference.index, index=reference.index.astype(str).str.upper())
    known_upper = known_upper[~known_upper.index.duplicated()]
    rows = known_upper.reindex(extra["gene"].str.upper())
    new_genes = extra.loc[rows.isna().to_numpy(), "gene"].drop_duplicates()
    reference = pd.concat(
        [reference, pd.DataFrame(0.0, index=new_genes.to_numpy(), columns=reference.columns)]
    )
    index = np.where(rows.isna().to_numpy(), extra["gene"].to_numpy(), rows.to_numpy())
    reference.loc[index] = 0.0
    for column, genes in pd.Series(index).groupby(extra["structure"].to_numpy()):
        reference.loc[genes.to_numpy(), column] = 1.0
    return reference

def _load_reference_scores(
    gold_data: Optional[Union[str, pd.DataFrame]],
    annotation_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Gene x compartment reference scores, indexed by upper-case symbol.

    Uses the same sources and hold-out option as RAP-X: a user table when
    given, otherwise bundled APEX-seq merged with RNALocate.
    """
    if gold_data is not None:
        table = (
            gold_data.copy()
            if isinstance(gold_data, pd.DataFrame)
            else (
                pd.read_parquet(gold_data)
                if str(gold_data).lower().endswith((".parquet", ".pq"))
                else pd.read_csv(gold_data)
            )
        )
        if "Common_Gene" not in table.columns:
            table = table.reset_index()
        if "Common_Gene" not in table.columns:
            raise ValueError("gold_data must contain a 'Common_Gene' column")
        reference = merge_gold(table, table.iloc[0:0], fuse="max")
        # Same per-gene 0-1 scaling RAP-X applies to a user table.
        reference = reference.sub(reference.min(axis=1), axis=0)
        reference = reference.div(reference.max(axis=1).replace(0, 1.0), axis=0)
    else:
        gold_apex, rna_locate = _load_default_gold_tables(
            apex_omm_specificity_filter=bool(
                annotation_cfg.get("apex_omm_specificity_filter", True)
            ),
            apex_omm_min_log2fc=float(annotation_cfg.get("apex_omm_min_log2fc", 0.75)),
            apex_omm_min_delta=float(annotation_cfg.get("apex_omm_min_delta", 0.50)),
        )
        species = annotation_cfg.get(
            "rnalocate_species", ("Homo sapiens", "Mus musculus")
        )
        if isinstance(species, str):
            species = tuple(v.strip() for v in species.split(",") if v.strip())
        from_rnalocate = _configure_rnalocate_augmentation(
            project_l1_to_seven(
                build_gold_from_rnalocate_l1(
                    rna_locate,
                    species=tuple(species) if species is not None else None,
                    agg_mode="prob_or",
                    score_floor=0.6,
                    use_pubmed_weight=True,
                ),
                mode="strict",
                mito_non_omm=None,
            ),
            augment_er=bool(annotation_cfg.get("rnalocate_augment_er", False)),
        )
        reference = _apply_extra_reference_markers(
            merge_gold(gold_apex, from_rnalocate, fuse="max")
        )
    reference.index = reference.index.astype(str).str.upper()
    reference = reference.loc[~reference.index.duplicated()]
    exclude_file = annotation_cfg.get("reference_exclude_genes_file")
    if exclude_file:
        with open(str(exclude_file), "r", encoding="utf-8") as handle:
            excluded = {line.strip().upper() for line in handle if line.strip()}
        kept = ~reference.index.isin(excluded)
        _logging.getLogger("cellscoper").info(
            "Reference hold-out: excluded %d of %d reference genes (%d listed)",
            int((~kept).sum()),
            int(kept.size),
            len(excluded),
        )
        reference = reference.loc[kept]
    return reference

def _configure_rnalocate_augmentation(
    gold_from_rna: pd.DataFrame,
    augment_er: bool = False,
) -> pd.DataFrame:
    """Keep broad RNALocate ER associations from overriding APEX specificity."""
    out = gold_from_rna.copy()
    if not augment_er:
        for column in ("ER", "ERM", "ER_Lumen"):
            if column in out.columns:
                out[column] = 0.0
    return out
