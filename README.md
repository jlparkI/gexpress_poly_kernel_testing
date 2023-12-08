# gexpress_poly_kernel_testing

This repo contains code needed to reproduce some key experiments for
fitting approximate and exact polynomials to the gene expression data
from Shen et al. 2023.

To use this repo, install the requirements in an active virtual environment,
then run:
```
python run_key_experiments.py
```

If run without any arguments, a list of options and required arguments
will be displayed. Use this CLI to reproduce the main experiments from
the paper. The shell scripts under shell_scripts provide examples.

The notebook code used to select an important motif subset is under
notebooks.

Note that cupy-cuda11x was originally used -- you may need to substitute
a different version of cupy depending on your cuda version.

All experiments involving approximate polynomial kernels used xGPR version 0.2
but later versions should give equivalent results.
