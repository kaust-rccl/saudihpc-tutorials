"""Exercise 3: write the TrainJob manifest for the CropGuard training. Defaults are for the T4 cluster (Harbor image, Kueue queue `batch`, PVCs `data` -> /data and `home` -> /home/jovyan).

    python exercise3/make_trainjob.py --namespace <ns> --gpus 2 | kubectl apply -f -           # 2 GPUs on one node (the default and the recommended form)
    python exercise3/make_trainjob.py --namespace <ns> --gpus 1 --epochs 2 | kubectl apply -f -  # 1 GPU: the plumbing test and the baseline to compare with
    python exercise3/make_trainjob.py --namespace <ns> --gpus 2 --out trainjob.yaml              # just write the file and read it

THE TWO GPUs MUST BE ON THE SAME NODE. How the manifest guarantees it:
  * --nodes 1 (default): both GPUs belong to ONE pod (`numNodes: 1`, `numProcPerNode: 2`, `nvidia.com/gpu: 2`), and a pod always sits on a single node.
  * --nodes N > 1 (e.g. 2 pods with 1 GPU each): a required pod-affinity rule on `kubernetes.io/hostname` makes every worker of this job land on the same node, or wait.
  * Either way the workers report their node name (NODE_NAME) and train_ddp.py stops with a clear message if they ever differ.

Why `runtimePatches`: a participant may create TrainJobs but not their own TrainingRuntimes, so the volumes, the Istio opt-out, the Kueue label and the affinity
travel inside the TrainJob. The shared runtime `torch-distributed` supplies torchrun and the rank variables.
The scripts are read from the repository clone in the home volume: <repo-dir>/exercise3/train_ddp.py.
"""
import argparse, shlex, sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--name', default='cropguard-train', help='TrainJob name; must be new each time (add a suffix or delete the old job)')
ap.add_argument('--namespace', default=None, help='default: the namespace this notebook runs in, else <your-namespace>')
ap.add_argument('--image', default='registry.tutorial.internal/ghcr/kubeflow/kubeflow/notebook-servers/jupyter-pytorch-cuda-full:v1.11.0')
ap.add_argument('--runtime', default='torch-distributed')
ap.add_argument('--queue', default='batch', help="Kueue queue name ('' = no queue label)")
ap.add_argument('--gpus', type=int, default=2, help='GPUs per node (0 = a CPU-only smoke test)')
ap.add_argument('--nodes', type=int, default=1, help='pods; more than 1 adds the same-node affinity rule')
ap.add_argument('--data-pvc', default='data'); ap.add_argument('--home-pvc', default='home'); ap.add_argument('--data-subpath', default='', help='mount only this folder of the data volume at /data')
ap.add_argument('--repo-dir', default='/home/jovyan/saudihpc-tutorials/kubeflow-tutorial', help='where the repository is cloned (in the home volume)')
ap.add_argument('--data-dir', default='/data/cropguard'); ap.add_argument('--out-dir', default='/data/models/cropguard')
ap.add_argument('--run-name', default=None, help='name of the model folder (default: timestamp)')
ap.add_argument('--lr', default='0.0003'); ap.add_argument('--epochs', default='2'); ap.add_argument('--seed', default='42'); ap.add_argument('--batch-size', default='64')
ap.add_argument('--procs', type=int, default=None, help='training processes per node (default: one per GPU)')
ap.add_argument('--extra', default='', help='more arguments for train_ddp.py, e.g. "--limit-train 1280"')
ap.add_argument('--env', action='append', default=[], metavar='KEY=VALUE', help='extra environment variable for the workers, e.g. --env NCCL_DEBUG=INFO or --env NCCL_P2P_DISABLE=1 (repeatable)')
ap.add_argument('--out', default=None, help='write here instead of stdout')
a = ap.parse_args()

ns = a.namespace
if ns is None:
    f = Path('/var/run/secrets/kubernetes.io/serviceaccount/namespace'); ns = f.read_text().strip() if f.exists() else '<your-namespace>'
q = {'kueue.x-k8s.io/queue-name': a.queue} if a.queue else {}
procs = a.procs or max(a.gpus, 1)
train_args = ['--data', a.data_dir, '--out', a.out_dir, '--lr', str(a.lr), '--epochs', str(a.epochs), '--seed', str(a.seed), '--batch-size', str(a.batch_size)] + (['--name', a.run_name] if a.run_name else []) + shlex.split(a.extra)
data_mount = {'name': 'data', 'mountPath': '/data'} | ({'subPath': a.data_subpath} if a.data_subpath else {})
pod_spec = {'securityContext': {'fsGroup': 100},
            'volumes': [{'name': 'data', 'persistentVolumeClaim': {'claimName': a.data_pvc}}, {'name': 'home', 'persistentVolumeClaim': {'claimName': a.home_pvc}},
                        {'name': 'dshm', 'emptyDir': {'medium': 'Memory', 'sizeLimit': '2Gi'}}],
            'containers': [{'name': 'node', 'volumeMounts': [data_mount, {'name': 'home', 'mountPath': '/home/jovyan'}, {'name': 'dshm', 'mountPath': '/dev/shm'}]}]}
if a.nodes > 1:                                                  # the same-node condition for several pods: every worker of THIS job on one machine
    pod_spec['affinity'] = {'podAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': [{
        'labelSelector': {'matchLabels': {'jobset.sigs.k8s.io/jobset-name': a.name}}, 'topologyKey': 'kubernetes.io/hostname'}]}}


def y(obj, ind=0):                                              # a tiny YAML writer, so the script needs nothing but Python
    pad, out = ' ' * ind, []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v: out.append(f'{pad}{k}:'); out += y(v, ind + 2)
            else: out.append(f'{pad}{k}: {scalar(v)}')
    else:
        for v in obj:
            if isinstance(v, dict):
                lines = y(v, ind + 2); out.append(f'{pad}- {lines[0].lstrip()}'); out += lines[1:]
            else: out.append(f'{pad}- {scalar(v)}')
    return out
def scalar(v):
    if isinstance(v, bool): return 'true' if v else 'false'
    if isinstance(v, (int, float)): return str(v)
    if isinstance(v, dict): return '{}'
    if isinstance(v, list): return '[]'
    return '"' + str(v).replace('"', '\\"') + '"'

manifest = {
    'apiVersion': 'trainer.kubeflow.org/v1alpha1', 'kind': 'TrainJob',
    'metadata': {'name': a.name, 'namespace': ns} | ({'labels': q} if q else {}),
    'spec': {
        'runtimeRef': {'apiGroup': 'trainer.kubeflow.org', 'kind': 'ClusterTrainingRuntime', 'name': a.runtime},
        'trainer': {
            'image': a.image, 'numNodes': a.nodes, 'numProcPerNode': procs,
            **({'resourcesPerNode': {'limits': {'nvidia.com/gpu': str(a.gpus)}}} if a.gpus else {}),
            'command': ['torchrun', f'{a.repo_dir}/exercise3/train_ddp.py'], 'args': train_args,
            'env': [{'name': 'PYTHONUNBUFFERED', 'value': '1'}, {'name': 'NODE_NAME', 'valueFrom': {'fieldRef': {'fieldPath': 'spec.nodeName'}}}]
                   + [{'name': k, 'value': v} for k, v in (e.split('=', 1) for e in a.env)]},
        'runtimePatches': [{'manager': 'tutorial.example.org/cropguard-exercise3', 'trainingRuntimeSpec': {'template': {'spec': {'replicatedJobs': [{
            'name': 'node',
            'template': {'metadata': {'labels': q} if q else {}, 'spec': {'template': {
                'metadata': {'annotations': {'sidecar.istio.io/inject': 'false'}, **({'labels': q} if q else {})}, 'spec': pod_spec}}}}]}}}}]}}
if not q: manifest['spec']['runtimePatches'][0]['trainingRuntimeSpec']['template']['spec']['replicatedJobs'][0]['template'].pop('metadata')
where = ('both GPUs are in ONE pod, so they are on one node' if a.nodes == 1 else f'{a.nodes} pods are forced onto ONE node by the pod-affinity rule below')
header = [f'# CropGuard training, {a.nodes} node(s) x {a.gpus} GPU(s). Same-node condition: {where}; the workers also check NODE_NAME at start-up.',
          f'# Scripts are read from {a.repo_dir} (home volume); the dataset is {a.data_dir} and the model is saved under {a.out_dir} (data volume).']
text = '\n'.join(header + y(manifest)) + '\n'
if a.out: Path(a.out).write_text(text); print(f'wrote {a.out}', file=sys.stderr)
else: sys.stdout.write(text)
