# Kubeflow tutorial: food-security story (work in progress)

Content for a Kubeflow tutorial built around the data-science model lifecycle: interactive computing, Kubeflow Pipelines, Katib,
distributed training and KServe. **Status: beta storyline in notebooks; no Kubeflow manifests yet.**

**Start with [`memory.md`](memory.md)**: what was tried, the results so far (with their caveats), open questions and next steps.

| Path | What |
|---|---|
| `notebooks/cropguard_beta.ipynb` | Main story: leaf-disease triage, PlantVillage (train) + RoCoLe (real-world check). Runs on CPU; section 9 uses a GPU if present |
| `notebooks/cropwatch_beta.ipynb` | Earlier GFSAD cropland story (reference only; needs Earth Engine) |
| `experiments/` | Scripts for the augmentation sweep, locked-half evaluation and the backbone comparison (they read the notebook's cache) |
| `gfsad/` | Scripts from the GFSAD exploration |
| `docs/agenda.csv` | Tutorial agenda |

## Quick start
```bash
conda create -n kf-tutorial-26 python=3.13 -y && conda activate kf-tutorial-26
# install the CUDA build of torch/torchvision for your machine first (https://pytorch.org), then:
pip install -r requirements.txt
python -m ipykernel install --user --name kf-tutorial-26 --display-name "Python (kf-tutorial-26)"
cd notebooks && jupyter lab cropguard_beta.ipynb
```
The notebook downloads the datasets into `notebooks/data/raw/` if they are missing (about 4.4 GB; PlantVillage from Hugging Face, RoCoLe from Mendeley) and builds
`notebooks/data/cache/` on the first run. Datasets, caches and model artifacts are git-ignored.

## Data and licenses
- PlantVillage (`mohanty/PlantVillage` on Hugging Face): the card says CC BY-SA 3.0 (some mirrors say CC0); confirm before redistributing derived data.
- RoCoLe (Parraga-Alava et al., Data in Brief 2019, doi 10.17632/c5yvn32dzg.2): CC BY 4.0.
