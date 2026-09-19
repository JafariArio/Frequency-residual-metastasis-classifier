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
