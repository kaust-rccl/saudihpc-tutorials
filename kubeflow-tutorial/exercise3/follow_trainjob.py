"""Exercise 3: follow a TrainJob: status changes plus the live log of every worker, until it finishes.

    python exercise3/follow_trainjob.py cropguard-train [--namespace <ns>]

Works from a notebook (uses the service account of the notebook) or from a laptop with a kubeconfig. Exit code 0 = Complete, 1 = Failed or timed out.
"""
import argparse, sys, threading, time
from pathlib import Path
from kubernetes import client, config

ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--namespace', default=None); ap.add_argument('--timeout', type=int, default=3600)
a = ap.parse_args()
try: config.load_incluster_config()
except Exception: config.load_kube_config()
ns = a.namespace
if ns is None:
    f = Path('/var/run/secrets/kubernetes.io/serviceaccount/namespace')
    ns = f.read_text().strip() if f.exists() else config.list_kube_config_contexts()[1]['context'].get('namespace', 'default')
co, core = client.CustomObjectsApi(), client.CoreV1Api()
GVR = dict(group='trainer.kubeflow.org', version='v1alpha1', namespace=ns, plural='trainjobs')
streaming, lock = set(), threading.Lock()

def stream(pod):
    tag = pod.split('-')[-3] + '-' + pod.split('-')[-2] if pod.count('-') >= 3 else pod
    try:
        r = core.read_namespaced_pod_log(pod, ns, follow=True, _preload_content=False)
        for line in r: 
            with lock: print(f'[{tag}] {line.decode(errors="replace").rstrip()}', flush=True)
    except Exception as e:
        with lock: print(f'[{tag}] (log stream ended: {str(e)[:80]})', flush=True)

def state(tj):
    conds = {c['type']: c for c in tj.get('status', {}).get('conditions', []) if c.get('status') == 'True'}
    for k in ('Complete', 'Failed', 'Suspended'):
        if k in conds: return k, conds[k].get('message', '')
    return 'Running/Created', ''

t0, last, final = time.time(), None, None
while time.time() - t0 < a.timeout:
    try: tj = co.get_namespaced_custom_object(name=a.name, **GVR)
    except client.ApiException as e:
        print(f'cannot read TrainJob {a.name} in {ns}: {e.reason}'); sys.exit(1)
    st, msg = state(tj)
    pods = core.list_namespaced_pod(ns, label_selector=f'jobset.sigs.k8s.io/jobset-name={a.name}').items
    summary = f'{st} | pods: ' + ', '.join(f'{p.metadata.name.split("-", 1)[-1][-18:]}={p.status.phase}' for p in pods)
    if summary != last: print(f'{time.strftime("%H:%M:%S")} {summary}' + (f' | {msg}' if msg else ''), flush=True); last = summary
    for p in pods:
        if p.status.phase in ('Running', 'Succeeded', 'Failed') and p.metadata.name not in streaming:
            streaming.add(p.metadata.name); threading.Thread(target=stream, args=(p.metadata.name,), daemon=True).start()
    if st in ('Complete', 'Failed'): final = st; break
    time.sleep(3)
time.sleep(3)                                                   # let the log streams flush their last lines
print(f'TrainJob {a.name}: {final or "timed out"}'); sys.exit(0 if final == 'Complete' else 1)
