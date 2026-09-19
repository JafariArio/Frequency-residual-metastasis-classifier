# Reproducibility notes

The final reported model is the archived 10.96M implementation copied to
`src/model/train_final_model.py`.

Do not mix it with an earlier public implementation containing a multiscale tokenizer,
scale fusion, quality head, or different frequency/loss formulation unless such files are
explicitly marked as legacy and not used for the reported results.

After verifying this generated repository:
1. initialize Git,
2. configure Git LFS,
3. commit,
4. push,
5. record the new commit hash in the reviewer response.

<!-- COMPLETE_RESULT_PROVENANCE -->
## Complete file-level result provenance

The reproducibility artifacts corresponding to the final manuscript package are
listed in `reproducibility/RESULT_PROVENANCE.csv`, with SHA-256 checksums in
`reproducibility/SHA256SUMS.csv`.

The historical experiments predate the Git-tracked public snapshot. The public
reproducibility package is therefore anchored to the revision package beginning
at commit `5bef38a58bec9a2913c137e39d3948b315fe9b56` together with the file-level SHA-256 manifests. This
commit must not be described as the historical training-time commit.

