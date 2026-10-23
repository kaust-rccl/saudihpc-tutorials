Research scripts behind the numbers in `../memory.md`. They read the cache that `notebooks/cropguard_beta.ipynb` builds
(`notebooks/data/cache`, override with `CROPGUARD_CACHE`) and must be run from the `kubeflow-tutorial/` directory:

    python experiments/sweep2.py              # augmentation recipe sweep on the RoCoLe development half (round 2)
    python experiments/final_eval.py          # scores the pre-chosen recipe once on the locked half, with bootstrap
    python experiments/backbones.py resnet18 resnet50 efficientnet_b0 convnext_tiny dinov2_vits14   # backbone comparison (dev half only)

Results are written next to the scripts as `.pkl` (git-ignored). `synth_snippet.py` and `regimes_snippet.py` hold the augmentation code (the notebook embeds the same code).
`sweep.py` is round 1 of the sweep and `sweep2.py` builds on it. They have only been exercised on the original machine.
