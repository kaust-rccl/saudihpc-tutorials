"""Exercise 4: compile the CropGuard pipeline, start a run, and follow it until it finishes.

    python exercise4/run_pipeline.py                                   # from a notebook server: finds Kubeflow Pipelines and your token itself
    python exercise4/run_pipeline.py --param lr=0.0003 --param epochs=3 --param min_locked_auc=0.95     # try to make the gate block
    python exercise4/run_pipeline.py --no-wait                          # start the run and return; watch it in the Pipelines UI

From a laptop (or for testing): --host http://localhost:8888 --token-file <file with a token for audience pipelines.kubeflow.org>.
"""
import argparse, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P
from kfp import Client, compiler

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--param', action='append', default=[], metavar='KEY=VALUE', help='pipeline parameter (repeatable): data_dir, models_dir, name_prefix, lr, epochs, seed, min_lab_acc, min_locked_auc, max_regression')
ap.add_argument('--host', default=os.environ.get('KF_PIPELINES_ENDPOINT', 'http://ml-pipeline.kubeflow.svc.cluster.local:8888'))
ap.add_argument('--token-file', default=os.environ.get('KF_PIPELINES_SA_TOKEN_PATH', '/var/run/secrets/kubeflow/pipelines/token'))
ap.add_argument('--namespace', default=None); ap.add_argument('--experiment', default='cropguard'); ap.add_argument('--run-name', default=None)
ap.add_argument('--image', default=P.DEFAULT_IMAGE); ap.add_argument('--repo-dir', default=P.REPO_DIR); ap.add_argument('--data-pvc', default='data'); ap.add_argument('--home-pvc', default='home')
ap.add_argument('--queue', default='batch'); ap.add_argument('--file', default='cropguard_pipeline.yaml', help='where to write the compiled pipeline')
ap.add_argument('--no-wait', action='store_true'); ap.add_argument('--timeout', type=int, default=3600)
a = ap.parse_args()

ns = a.namespace
if ns is None:
    f = Path('/var/run/secrets/kubernetes.io/serviceaccount/namespace'); ns = f.read_text().strip() if f.exists() else None
compiler.Compiler().compile(P.build(a.image, a.repo_dir), a.file); P.add_platform_spec(a.file, a.data_pvc, a.home_pvc, a.queue)
print(f'compiled {a.file}')
types = {'lr': float, 'epochs': int, 'seed': int, 'min_lab_acc': float, 'min_locked_auc': float, 'max_regression': float}
args = {k: types.get(k, str)(v) for k, v in (p.split('=', 1) for p in a.param)}
token = Path(a.token_file).read_text().strip() if Path(a.token_file).exists() else None
client = Client(host=a.host, existing_token=token, namespace=ns)
run = client.create_run_from_pipeline_package(a.file, arguments=args, run_name=a.run_name or f'cropguard-{time.strftime("%m%d-%H%M")}', experiment_name=a.experiment, namespace=ns)
print(f'run started: {run.run_id}  parameters: {args or "(defaults)"}')
if a.no_wait: sys.exit(0)

seen, t0 = {}, time.time()
while time.time() - t0 < a.timeout:
    r = client.get_run(run.run_id); state = str(r.state)
    tasks = {t.display_name: str(t.state) for t in (getattr(r.run_details, 'task_details', None) or []) if getattr(t, 'display_name', None)}
    for k, v in tasks.items():
        if seen.get(k) != v: print(f'{time.strftime("%H:%M:%S")}  {k:14s} {v}', flush=True); seen[k] = v
    if state in ('SUCCEEDED', 'FAILED', 'CANCELED', 'SKIPPED'): print(f'run {state}'); sys.exit(0 if state == 'SUCCEEDED' else 1)
    time.sleep(5)
print('timed out waiting for the run'); sys.exit(1)
