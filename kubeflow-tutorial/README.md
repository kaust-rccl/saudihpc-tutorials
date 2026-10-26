# Operation CropGuard: a Kubeflow tutorial

A hands-on tutorial in five exercises that takes one machine-learning problem from a notebook to a running service on Kubeflow.

**The problem.** A farm app should tell a farmer whether a coffee leaf is sick from a phone photo. The best public training photos (PlantVillage) were taken in a lab; real field photos look different. You train a model on lab photos, find out how it does in the field, then use Kubeflow to tune it, scale it up, gate it, and serve it.

## Exercises

| # | Topic | Kubeflow component | What you do | Where |
|---|---|---|---|---|
| 1 | Workspace and baseline | Notebooks | Inspect your GPU workspace and data, train a ConvNeXt-Tiny baseline on lab photos, test it on coffee photos | [exercise1/](exercise1/) |
| 2 | Hyperparameter search | Katib | Run a Bayesian search over the learning rate with short trials | [exercise2/](exercise2/) |
| 3 | Scaling up the winner | Trainer (TrainJob) | Run the same training as a background job on 1 or 2 GPUs with PyTorch DDP; compare throughput | [exercise3/](exercise3/README.md) |
| 4 | A gate before production | Pipelines | Check data, train, evaluate, gate, publish; a weak model is blocked | [exercise4/](exercise4/README.md) |
| 5 | From a file to a service | KServe | Serve the `production` model over HTTP, with a tuned cut-off and a "refer to extension officer" fallback | [exercise5/](exercise5/README.md) |

Each exercise builds on the previous one: Exercise 3 writes a model folder, Exercise 4 evaluates and publishes it as `production`, and Exercise 5 serves whatever `production` points to.

## The data and the two rules that matter

Two datasets are packed into `/data/cropguard`:

* **PlantVillage** (lab leaves, 38 classes) is split **by leaf**, so sibling photos of one leaf never land in both train and test.
* **RoCoLe** (real coffee-leaf photos, healthy or sick) is split **by plant** into a `dev` half and a `locked` half.
  * `dev` is for making choices (a setting, a recipe, a cut-off).
  * `locked` is the final exam. Nothing is chosen with it. Only the Exercise 4 evaluate step scores it.

Models are judged by lab accuracy and by a ranking score (AUC) of a "how sick?" score on coffee photos. The dataset itself is not in this repository; see the next section.

## Layout

```
kubeflow-tutorial/
├── common/        cropguard_lib.py: shared data loading, model, scores, evaluation (used by exercises 3-5)
├── exercise1/     notebook: workspace and baseline
├── exercise2/     train_trial.py (one Katib trial) and k8s/katib-ui-bayes-aws-pvc.yaml
├── exercise3/     train_ddp.py, make_trainjob.py, follow_trainjob.py, sample TrainJob manifests
├── exercise4/     pipeline.py, pipeline_steps.py, run_pipeline.py
└── exercise5/     serve_cropguard.py, make_inferenceservice.py, call_service.py, sample manifest
```

## Prerequisites

* A Kubeflow cluster with Notebooks, Katib, Trainer, Pipelines and KServe (developed on Pipelines 2.16.1 and KServe 0.18 in RawDeployment mode).
* GPU nodes (written for NVIDIA T4s; developed on a single GB10). A Kueue queue named `batch` is assumed for GPU jobs; pass `--queue ''` to drop the label.
* A notebook server with a GPU, based on `jupyter-pytorch-cuda-full:v1.11.0` (Exercise 4 can use an image that adds `kfp-kubernetes`, but the pipeline compiles without it).
* Two volumes mounted in every pod:

  | Volume | Mount | Holds |
  |---|---|---|
  | `data` | `/data` | the dataset at `/data/cropguard`, and saved models at `/data/models/cropguard` |
  | `home` | `/home/jovyan` | this repository, cloned |

* `kubectl` access from the notebook terminal. Images are pulled from the cluster's Harbor registry; the default image paths are placeholders for the author's cluster, so override them with `--image` where your registry differs.

## Getting started

In a notebook terminal:

```bash
cd /home/jovyan
git clone <repository-url> saudihpc-tutorials
cd saudihpc-tutorials/kubeflow-tutorial
mkdir -p /data/models/cropguard
```

Then work through the exercises in order. Scripts are run from `kubeflow-tutorial/` and read from the home volume, so nothing is copied to the data volume or built into an image. Later: `git pull` to update.

* **Exercise 1:** open `exercise1/exercise1_workspace_and_baseline.ipynb` and run it top to bottom.
* **Exercise 2:** edit the namespace in `exercise2/k8s/katib-ui-bayes-aws-pvc.yaml`, then paste it into the Katib UI (New Experiment, "Edit and submit YAML contents"). The manifest runs `train_trial.py` from `/data/code/`, so copy that file there first.
* **Exercises 3 to 5:** follow the step-by-step instructions in each exercise's README.

## Status

Exercises 3 to 5 were written and tested on a dev cluster with one GPU and have **not yet been run on the T4 cluster**. Each README ends with a list of what was and was not tested, and what to check first. The KServe address from outside the cluster is the least verified part.

## Notes for contributors

* Datasets, model weights (`*.pt`), arrays and generated artifacts are git-ignored; do not commit them.
* Keep the locked coffee photos out of any training, tuning or search code.
