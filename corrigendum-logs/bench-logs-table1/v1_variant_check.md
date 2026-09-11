# Which program is the v1 comparator?

`legacy/goldbach_gpu3.cu` on `release/v2.0.1` is NOT the file named
`goldbach_gpu3` at tag v1.1.0. Content comparison by md5:

| Tag : path | lines | matches branch copy? |
|---|---|---|
| v2.0.1 : legacy/goldbach_gpu3.cu | 521 | (this is the one built) |
| v2.0.0 : legacy/goldbach_gpu3.cu | 521 | identical |
| v1.1.0 : legacy/goldbach_gpu3a.cu | 521 | differs by 1 line (header comment only) |
| v1.1.0 : src/goldbach_gpu3.cu | 736 | different program (multi-worker, multi-GPU) |
| v1.0.0 : src/goldbach_gpu3.cu | 521 | differs by 10 lines (comment + default SEG_SIZE/P_SMALL) |

So the branch copy is v1.1.0's `goldbach_gpu3a`, with only a comment changed.
The file called `goldbach_gpu3` at v1.1.0 is a different, larger program.

Measured to see whether the choice matters, 10^10, --gpus=1, same machine:

| Variant | mean | n |
|---|---|---|
| legacy/goldbach_gpu3 (built from the branch) | 13557.0 ms | 5 |
| v1.1.0 src/goldbach_gpu3 (multi-GPU variant) | 13690.5 ms | 3 |

1.0% apart, inside the run-to-run spread. The variant choice does not explain
the 24.9% gap against the published 18056.5 ms; that gap is environmental.
