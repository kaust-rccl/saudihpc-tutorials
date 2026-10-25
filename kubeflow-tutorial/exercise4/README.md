# Exercise 4: Kubeflow Pipelines, a gate between a model and production

**Status: written and tested on the dev cluster (Pipelines 2.16.1, one GB10 GPU). Not yet run on the T4 cluster.**

## The story
In Exercise 3 a person trained a model and saved it. Now ask: *how do we stop a weak model from reaching the farmers, and how do we do this every time new field photos arrive without anyone remembering the steps?*

The answer is a **pipeline**: the steps you have been running by hand, written down once and run the same way every time, each in its own pod, with a **gate** in the middle.

```
check data  ->  train (1 GPU)  ->  evaluate  ->  gate  ->  publish   (only if the gate says "passed")
                                      |            |
                         the FINAL EXAM:           floors + "not worse than the model in production"
                         the locked half of the
                         coffee photos, scored
                         once per model
```
* **check data**: all photos present, leaf and plant splits still clean (the same checks as Exercise 1).
* **train**: Exercise 3's training (2 epochs), on one GPU, saved as a *candidate*.
* **evaluate**: the only place the locked coffee photos are scored. It scores the candidate and the model currently in production on the same photos.
* **gate**: three rules: lab accuracy at least 0.95; real-world ranking score (locked AUC) at least 0.80; and not more than 0.03 below the model in production (0.03 is about the run-to-run noise, so we do not block on luck).
* **publish**: moves the `production` link to the candidate. The old model stays on disk, so rolling back is one command. A blocked candidate stays on disk too, for inspection.

**The point to land:** a model reaches production only by passing the same exam as every other, and the pipeline's graph, parameters, logs and verdicts are saved for every run. Try `min_locked_auc=0.95` and watch the gate block (the *publish* box turns grey: skipped). Try different seeds: two identical recipes differ by a few hundredths, which is why the regression rule has a tolerance.

## Layout
Both volumes are mounted in every step: `data` at `/data` (dataset, models) and `home` at `/home/jovyan`, where the repository is cloned. Every step runs a script from
`/home/jovyan/saudihpc-tutorials/kubeflow-tutorial/` (see Exercise 3 for how the code gets there).

| File | What |
|---|---|
| `pipeline.py` | defines the pipeline and compiles it (`python exercise4/pipeline.py --out cropguard_pipeline.yaml`) |
| `run_pipeline.py` | compiles, submits a run, and prints each step's state until the end |
| `pipeline_steps.py` | the work of check-data, evaluate, gate and publish (also runnable by hand) |
| `../exercise3/train_ddp.py`, `../common/cropguard_lib.py` | training and shared helpers |
| `image/` | the Dockerfile and build script of the notebook image that includes `kfp-kubernetes` (below) |

Every step is a "container component": Kubeflow starts a pod and runs one of our scripts. **Nothing is installed in the step image.** The volume mounts and the Kueue label are written into the compiled file by `add_platform_spec()`
in `pipeline.py` with plain PyYAML; I checked that its output is **identical** to the `kfp-kubernetes` package's, so the pipeline compiles with or without the package.

## The notebook image with `kfp-kubernetes`
The stock image (`jupyter-pytorch-cuda-full:v1.11.0`) does not include `kfp-kubernetes`. A copy with it added is published as **`mshaikh/kubeflow-demo:kfp-torch-v1110`** (Docker Hub, public, linux/amd64 and linux/arm64).
It is the stock image plus one 128 KB layer: the package's files are copied into site-packages (it is pure Python and its dependencies, `kfp` 2.16.1 and `protobuf` 6.x, are already there), so the build runs no code and one Dockerfile serves both architectures.
Verified from a fresh pull: `from kfp import kubernetes` works, `kfp` 2.16.1, `kfp-kubernetes` 2.16.1, torch 2.12.0. Rebuild or change the version: `bash exercise4/image/build_and_push.sh` (needs `docker login`).
**Your Harbor policy:** images must come from Harbor, so pull this one through your Harbor's Docker Hub proxy (the path depends on how Harbor is set up) or mirror it there. For the pipeline *steps*, `--image` sets the image (default: the Harbor copy of the stock image).

## Try the steps by hand first (2 minutes, in a terminal, from the repository folder)
```bash
M=/data/models/cropguard
python exercise4/pipeline_steps.py check-data --data /data/cropguard --models-dir $M --run-name t4-1gpu
python exercise4/pipeline_steps.py evaluate   --data /data/cropguard --models-dir $M --run-name t4-1gpu     # needs a model folder from Exercise 3
python exercise4/pipeline_steps.py gate       --models-dir $M --run-name t4-1gpu
python exercise4/pipeline_steps.py publish    --models-dir $M --run-name t4-1gpu                            # creates $M/production
```

## Run the pipeline on the T4 cluster
```bash
# from a notebook server created with "access to Kubeflow Pipelines" ticked (it mounts your token); otherwise see point 1 below
python exercise4/run_pipeline.py                                    # defaults: 2 epochs, lr 3e-4, seed 42
python exercise4/run_pipeline.py --param seed=7                     # a different candidate: the gate now compares it with the model in production
python exercise4/run_pipeline.py --param min_locked_auc=0.95        # an impossible bar: the gate blocks, publish is skipped
python exercise4/run_pipeline.py --no-wait                          # start it and watch the graph in the Pipelines UI instead
```
Defaults assume the T4 layout: `data` at `/data` (dataset `/data/cropguard`, models `/data/models/cropguard`), `home` at `/home/jovyan`, Harbor step image, Kueue queue `batch` on the two GPU steps.
Other layouts: `--data-pvc`, `--home-pvc`, `--image`, `--queue ''`, `--repo-dir`, `--param data_dir=...`, `--param models_dir=...`, `--host`, `--token-file`, `--namespace` (see `--help`).
Pipeline parameters: `data_dir, models_dir, name_prefix, lr, epochs, seed, min_lab_acc, min_locked_auc, max_regression`.
Each run's model folder is `<models_dir>/<name_prefix>-<run id>/`; its records (data report, evaluation, gate verdict) are in `<models_dir>/_pipeline/<same name>/`.

## What was tested here, and what was not
**Tested (dev cluster):** compile; full runs of about 3 minutes (check data, train 2 epochs, evaluate, gate, publish: all succeeded, `production` moved, records written), the last one from the home-volume layout with both volumes mounted;
a run with `min_locked_auc=0.95` where the gate blocked, the publish branch showed SKIPPED, and `production` did not move; a run with another seed that was compared with the model in production, passed all three rules and replaced it
(and the Exercise 5 service, already running, switched to it by itself); the steps by hand; the image with `kfp-kubernetes` (see above).
**Bugs found and fixed on the way (worth knowing):** `dsl.PIPELINE_JOB_ID_PLACEHOLDER` is NOT substituted inside container arguments; it must be passed to each step as an input parameter (done: `run_id`).
Nested `ConcatPlaceholder` expressions in arguments were passed through as literal text, so every step builds its own paths from plain arguments instead.
**Not tested, check first on the T4 cluster:**
1. **Token and endpoint from the notebook.** `run_pipeline.py` reads the token from `/var/run/secrets/kubeflow/pipelines/token` (or `$KF_PIPELINES_SA_TOKEN_PATH`) and talks to `http://ml-pipeline.kubeflow.svc.cluster.local:8888`. If your notebooks have neither, create a token with `kubectl create token default-editor -n <ns> --audience pipelines.kubeflow.org` and pass `--token-file`.
2. **The Kueue label on pipeline pods.** KFP runs steps as plain Pods through Argo, not as Jobs. The label is added to the two GPU steps; whether your policy (and Kueue) accepts or needs it for Pods is unknown. `--queue ''` shows whether the label is the problem.
3. **GPU-only requests:** the GPU steps ask for `nvidia.com/gpu: 1` and nothing else. The CPU steps ask for nothing.
4. **Image pulls** for the step pods and the KFP launcher from Harbor.
5. **Two GPU steps per run** (train, then evaluate) means two waits in a busy queue.
6. The training step uses `--workers 0` on purpose: pipeline pods have only 64 MB of shared memory, which the data-loader processes would overflow.
