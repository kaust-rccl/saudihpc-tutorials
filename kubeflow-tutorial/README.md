# Kubeflow tutorial: food-security story (work in progress)

Content for a Kubeflow tutorial built around the data-science model lifecycle: interactive computing, Kubeflow Pipelines, Katib,
distributed training and KServe. **Status: beta storyline in notebooks; no Kubeflow manifests yet.**

**Start with [`memory.md`](memory.md)**: what was tried, the results so far (with their caveats), open questions and next steps.

| Path | What |
|---|---|
| `notebooks/cropguard_beta.ipynb` | Main story: leaf-disease triage, PlantVillage (train) + RoCoLe (real-world check). Runs on CPU; section 9 uses a GPU if present |
| `notebooks/exercise1_workspace_and_baseline.ipynb` | **Exercise 1 (student notebook)**: GPU check, data volume, fine-tune ConvNeXt-Tiny on lab photos, field test on RoCoLe, manual augmentation fix. Needs a GPU and the packed dataset below; no downloads |
| `data_prep/build_cropguard_dataset.py` | Instructor-side: packs the dataset for the participants' PVC (1.55 GB: images, split tables, ConvNeXt-Tiny weights) |
| `exercise2/k8s/katib-ui-bayes.yaml` | **Exercise 2 (Katib), stable experiment**: Bayesian search over the learning rate, 4 trials, ran in the Katib UI in 3 min 17 s. Paste it into "Edit and submit YAML". It expects the ConfigMap `cropguard-train-trial` (the trial script) and the dataset volume in the namespace |
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

### Exercise 1 (student notebook)
```bash
python data_prep/build_cropguard_dataset.py          # from kubeflow-tutorial/; downloads the raw archives if missing, writes data/cropguard/ (~1 min)
cd notebooks && jupyter lab exercise1_workspace_and_baseline.ipynb
```
On the cluster, copy `data/cropguard/` onto the PVC. The notebook looks for the folder containing `MANIFEST.json` at `$CROPGUARD_DATA`, `/data/cropguard`,
`/home/jovyan/data/cropguard`, `../data/cropguard` or `data/cropguard` (edit `CANDIDATES` in step 2 if the PVC is mounted elsewhere). It saves models to `$CROPGUARD_OUT` (default `~/cropguard-artifacts`).

## Data and licenses
- PlantVillage (`mohanty/PlantVillage` on Hugging Face): the card says CC BY-SA 3.0 (some mirrors say CC0); confirm before redistributing derived data.
- RoCoLe (Parraga-Alava et al., Data in Brief 2019, doi 10.17632/c5yvn32dzg.2): CC BY 4.0.
