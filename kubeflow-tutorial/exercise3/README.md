# Exercise 3: Kubeflow Trainer, scaling up the winner

**Status: written and tested on the dev cluster (one GB10 GPU). Not yet run on the T4 cluster.** The "untested" list at the bottom is what to look at first.

## The story
Exercise 2 found good settings with many tiny trials. Now we spend real compute on the winner: **the same training as a background batch job on 2 GPUs**.
You submit it, watch the live logs of every worker, and the finished model is written to your volume together with a small `metadata.json`
(classes, settings, scores, and the cut-off tuned on the dev half). Exercise 4 will test that model; Exercise 5 will serve it.

Be honest with participants about what scaling buys. The training is 2 epochs, the length the Katib winner used, and it reaches the quality of the Katib trials
(lab accuracy about 97%, dev ranking score about 0.82 to 0.87, which varies by seed). More epochs did not help on this dataset. Scaling up is about **throughput and workflow**
(more GPUs, background execution, logs, saved artifacts), not about a better model. The number to compare is photos per second on 1 and on 2 GPUs.

## Layout (the same in every exercise)
Two volumes are mounted in every pod: `data` at `/data` (the dataset `/data/cropguard`, and `/data/models/cropguard` where models are saved) and `home` at `/home/jovyan`.
The repository is cloned into the home volume, and every script runs from there:
`/home/jovyan/saudihpc-tutorials/kubeflow-tutorial/exercise<N>/<script>.py` (shared helpers: `common/cropguard_lib.py`). Nothing is copied to the data volume and nothing is built.

| File | What |
|---|---|
| `train_ddp.py` | the training script: one process or `torchrun` on N GPUs (PyTorch DDP); saves the model; refuses to run across nodes; never scores the locked half |
| `make_trainjob.py` | writes the TrainJob manifest (`... | kubectl apply -f -`) |
| `follow_trainjob.py` | follows status and the live log of every worker until the job ends |
| `trainjob.yaml` | a sample manifest with the T4 defaults (2 GPUs), to read |

## Get the code into the home volume (once)
In a notebook terminal: `cd /home/jovyan && git clone <repository> saudihpc-tutorials` (later: `git pull`). **These folders are not committed yet**, so until they are pushed use the copy helper from a machine with `kubectl` access:
`bash tools/copy_to_notebook.sh <namespace> <notebook-pod>`. Also: `mkdir -p /data/models/cropguard`.

## The same-node rule (the two GPUs must be on one node)
* **Default (`--nodes 1 --gpus 2`)**: both GPUs belong to one pod (`numNodes: 1`, `numProcPerNode: 2`, `nvidia.com/gpu: 2`). A pod always sits on one node, so this is guaranteed. If no node has 2 free GPUs the job simply waits.
* **`--nodes 2 --gpus 1`** (two pods): the manifest carries a *required pod-affinity rule* on `kubernetes.io/hostname` for this job's own pods, so both land on the same node or wait.
* **Safety net**: the workers receive `NODE_NAME` (the node they run on) and `train_ddp.py` stops with a clear message if the node names differ (`--allow-multi-node` turns that off).
The first lines of a generated manifest say which of the two applies.

## Test on the T4 cluster (suggested order)
From `/home/jovyan/saudihpc-tutorials/kubeflow-tutorial`. The namespace is read from the notebook itself, so `--namespace` is only needed from outside.

**1. The 1-GPU benchmark (2 epochs).** Proves image, volumes, queue, logs and saving, and gives the baseline for the comparison.
```bash
python exercise3/make_trainjob.py --gpus 1 --name cropguard-1gpu --run-name t4-1gpu | kubectl apply -f -
python exercise3/follow_trainjob.py cropguard-1gpu
```
**2. The 2-GPU run on one node.**
```bash
python exercise3/make_trainjob.py --gpus 2 --name cropguard-2gpu --run-name t4-2gpu | kubectl apply -f -
python exercise3/follow_trainjob.py cropguard-2gpu
```
**3. Optional: two pods with 1 GPU each (the affinity rule keeps them on one node).**
```bash
python exercise3/make_trainjob.py --nodes 2 --gpus 1 --name cropguard-2pods --run-name t4-2pods | kubectl apply -f -
```
**4. Compare.**
```bash
python - <<'E'
import json, glob
for f in sorted(glob.glob('/data/models/cropguard/t4-*/metadata.json')):
    m = json.load(open(f)); t = m['training']; s = m['scores']
    print(f"{m['name']:10s} gpus {t['gpus']} | {t['epochs']} epochs | {t['train_seconds']:6.1f} s | {t['mean_photos_per_second']:6.1f} photos/s | lab test acc {s['test_acc']:.3f} | dev ranking score {s['dev_auc']:.3f} | cut-off {s['cutoff']:.2f}")
E
```
**What the dev cluster gave (one GB10, 1 GPU, 2 epochs, run as a TrainJob):** 33.9 s of training (epoch 1: 16.1 s, epoch 2: 14.3 s), 39.6 s for the whole job, about 300 photos per second, lab test accuracy 0.966, dev ranking score 0.823.
With your measured factor (GB10 is about 1.3x a T4: 14 s against 19 s per epoch), expect about 19 s per epoch on one T4, so roughly 45 to 50 s of training and about a minute for the whole job. With 2 T4s ideally half the training time (about 25 s); the first epoch also pays a few seconds of start-up.

**Useful while it runs:** `kubectl get trainjob,jobset,pods -n <ns>`, `kubectl describe trainjob <name> -n <ns>`, `kubectl get workloads -n <ns>` (the Kueue queue), `kubectl get pods -o wide` (which node).
Delete a finished job before reusing its name: `kubectl delete trainjob <name> -n <ns>`.

## Options worth knowing
`make_trainjob.py --help`. Most useful: `--queue ''` (no Kueue label), `--image`, `--data-pvc`, `--home-pvc`, `--repo-dir`, `--epochs`, `--procs`, `--extra "--limit-train 1280"` (a quick smoke test),
`--env NCCL_DEBUG=INFO` and `--env NCCL_P2P_DISABLE=1` (two-GPU communication problems). Training options (`train_ddp.py --help`): `--lr`, `--epochs`, `--batch-size` (per GPU), `--freeze-stages`, `--limit-train`.
The learning rate is not rescaled for 2 GPUs; the batch per GPU stays 64, so the global batch is 128 and each epoch has half as many steps.

## What was tested here, and what was not
**Tested (dev cluster, 1 GPU):** the manifest is accepted by the Trainer admission webhook (including the pod-affinity form); a 1-GPU TrainJob ran end to end from the home-volume layout (pod, live log, model and `metadata.json` written, status Complete: the benchmark above);
a 2-process launch inside one pod through the Trainer's `torchrun` (gloo, sharing the GPU) trained and saved; **a 2-pod job with the affinity rule** scheduled, found each other, and reported `all 2 workers run on node ...`;
the node check refuses when ranks report different node names (simulated) and passes when they agree.
**Not tested, check first on the T4 cluster:**
1. **NCCL between 2 real GPUs.** If it hangs or errors, rerun with `--env NCCL_DEBUG=INFO`, then `--env NCCL_P2P_DISABLE=1`.
2. **The Kueue queue** (`kueue.x-k8s.io/queue-name: batch`): the label is on the TrainJob and on the generated Job and Pods; which of these your admission policy checks, and whether Kueue admits a TrainJob, is unknown.
3. **The Harbor image path** is copied from your Katib file; the Trainer runtime's own image is overridden with it.
4. **GPU-only requests**: the job asks for `nvidia.com/gpu` and nothing else (the queue covers only GPUs).
5. **Volumes**: `data` is read-write at `/data` (the model is saved there) and `home` at `/home/jovyan` (the scripts are read from there). Two pods on one node can share a ReadWriteOnce volume; they could not on two nodes, which is another reason for the same-node rule.
6. The `runtimePatches` manager name `tutorial.example.org/cropguard-exercise3` is arbitrary but must be domain-prefixed (the validator insists).
