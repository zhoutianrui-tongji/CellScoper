"""Column and representation names shared by the pipeline modules."""


METASPOT_COL = "metaspot"  # MetaSpot id column (cell id + local index)
METASPOT_LOCAL_COL = "metaspot_in_cell"
LENS_REP = "X_lens"  # LENS embedding z (32-d); its diagnostics, decoder and null are in adata1.uns["lens"]

# 转录本表中行数巨大但取值重复的文字列：运行时用 categorical 存（每行一个整数），写 Parquet 时还原成普通文字
COMPACT_STRING_COLS = (
    "cell_id", METASPOT_COL, "subcellular_broad_compartment", "subcellular_annotation_placed", "subcellular_annotation",
    "subcellular_annotation_validated", "subcellular_annotation_exploratory", "subcellular_annotation_support",
    "subcellular_annotation_domain",
)

# 注释、细胞 x 结构表实际用到的转录本列；其余列留在 module5_final_data.parquet，写结果时再分批读回
TRANSCRIPT_WORKING_COLS = (
    "cell_id", METASPOT_COL, "feature_name", "x_location", "y_location", "z_location", "in_nucleus", "meta-domain",
)
