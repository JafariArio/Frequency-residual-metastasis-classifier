# Reproducibility

This repository contains the implementation and analysis files corresponding to
the revised manuscript. Earlier developmental components that were not used to
produce the final reported results are not part of the reproducibility workflow
listed below.

## Result-code provenance

Code commit recorded when this reproducibility package was prepared:

`5bef38a58bec9a2913c137e39d3948b315fe9b56`

Expected manuscript-result commit prefix:

`5bef38a`

If these identifiers differ, verify the provenance before stating that the
expected commit generated the reported results.

## Reproducibility artifacts

| Purpose | Repository file | SHA-256 |
|---|---|---|


The machine-readable provenance table is available at
`reproducibility/RESULT_PROVENANCE.csv` and the checksum manifest at
`reproducibility/SHA256SUMS.csv`.

## Recommended verification

1. Confirm that the checkpoint and canonical prediction files are the exact
   artifacts used for the manuscript results.
2. Confirm that the configuration corresponds to the frozen final model.
3. Run the DeLong and paired-bootstrap scripts from the canonical prediction files.
4. Run the CAMELYON17-WILDS inference pipeline with the frozen checkpoint.
5. Run SmoothGrad with the same frozen checkpoint and preprocessing pipeline.

## Development-code note

The repository previously contained experimental/developmental components.
Only the files identified in this document and the provenance table should be
used to reproduce the revised manuscript results.
