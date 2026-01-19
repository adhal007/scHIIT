suppressPackageStartupMessages({
  library(scGeneXpress)
  library(anndata)
  library(Matrix)
  library(dplyr)
})

# ---- paths ----
brain_path <- '/mnt/lscratch/users/adhal/SingleCellUtils/analysis_notebooks/sead_sampled_10K_171125.h5ad'
out_dir    <- "/mnt/lscratch/users/adhal/SingleCellUtils/outputs/scGX/tier1_SEAD/"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
adata <- read_h5ad(brain_path)
adata <- adata[adata$obs$disease == 'normal', ]
print("Data Read")
# extract counts and metadata
# (as you wrote)
counts <- t(as.matrix(adata$raw$X))

# # ---- minimal fix: ensure UNIQUE gene rownames and aligned cell IDs ----
# genes0 <- if ("feature_name" %in% colnames(adata$var)) as.character(adata$var$feature_name) else rownames(adata$var)
# genes0[is.na(genes0) | genes0 == ""] <- rownames(adata$var)[is.na(genes0) | genes0 == ""]
print(length(counts))
print(length(adata$var$feature_name))
rownames(counts) <- adata$var$feature_name       # gene names
colnames(counts) <- rownames(adata$obs)          # cell IDs

# cell_ids <- rownames(adata$obs)
# # # (only make unique if needed)
# # if (anyDuplicated(cell_ids)) cell_ids <- make.unique(cell_ids, sep = "_cell")
# colnames(counts) <- cell_ids
# print(length(cell_ids))
# metadata
df <- adata$obs
meta <- data.frame(`Cell ID` = rownames(df),
                   `Cluster Name` = df$Class,
                   check.names = FALSE)
colnames(meta) <- c('Cell ID', 'Cluster Name')

# output settings

resName <- "neuronal_broad_type"
pb = "/mnt/lscratch/users/adhal/scrna_target_idf_v3/data/Ref_sNucSymbol"

print('Running sc Gene Express')

# Run scGeneXpress
run_scGeneXpress(
  data       = counts,
  metadata   = meta,
  org        = "human",
  dir        = out_dir,
  file.name  = resName,
  precBack = pb,
  discretize = TRUE,
  sig.frames = TRUE,
  ncores     = 4,
  fixseed    = 12345
)