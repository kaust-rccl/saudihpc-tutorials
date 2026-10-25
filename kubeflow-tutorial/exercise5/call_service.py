"""Exercise 5: call the CropGuard service the way a phone app would: photo in, answer out.

    python exercise5/call_service.py                                                         # from a notebook: the service in your own namespace, inside the cluster, no login
    python exercise5/call_service.py --cluster aws.mlops101.io --token-file my_token           # from outside, through the cluster address (the gateway needs a login)
    python exercise5/call_service.py --url http://localhost:8080 --data data/cropguard         # any address, e.g. a service you port-forwarded

Sends a few real photos from the dataset (a lab leaf, healthy and sick coffee leaves from the dev half) plus two awkward inputs, and prints the answers.
Standard library plus numpy and Pillow only.
"""
import argparse, base64, io, json, sys, time, urllib.request, urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--url', default=None, help='base address of the service, without /v1/...  (default: the in-cluster address of the service in your namespace)'); ap.add_argument('--model', default='cropguard')
ap.add_argument('--cluster', default=None, help='the cluster address, e.g. aws.mlops101.io: calls https://<cluster>/serving/<namespace>/<model>'); ap.add_argument('--namespace', default=None); ap.add_argument('--scheme', default='https')
ap.add_argument('--token', default=None, help='a login token for the cluster gateway (sent as Authorization: Bearer ...)'); ap.add_argument('--token-file', default=None); ap.add_argument('--cookie', default=None, help='the browser session cookie of a logged-in Kubeflow user, e.g. "oauth2_proxy_kubeflow=..."')
ap.add_argument('--path-template', default='/serving/{namespace}/{name}'); ap.add_argument('--print-url', action='store_true', help='show the address that would be called and stop')
ap.add_argument('--data', default='/data/cropguard'); ap.add_argument('--repeat', type=int, default=10, help='extra single-photo calls to time')
a = ap.parse_args()
ns = a.namespace
if ns is None:
    f = Path('/var/run/secrets/kubernetes.io/serviceaccount/namespace'); ns = f.read_text().strip() if f.exists() else None
if a.url is None:
    if a.cluster: a.url = f'{a.scheme}://{a.cluster}' + a.path_template.format(namespace=ns or '<namespace>', name=a.model)
    elif ns: a.url = f'http://{a.model}-predictor.{ns}.svc.cluster.local'
    else: sys.exit('give --url or --cluster (and --namespace): cannot tell which service to call')
if a.print_url: sys.exit(f'{a.url.rstrip("/")}/v1/models/{a.model}:predict')
HEADERS = {}
token = a.token or (Path(a.token_file).read_text().strip() if a.token_file else None)
if token: HEADERS['Authorization'] = f'Bearer {token}'
if a.cookie: HEADERS['Cookie'] = a.cookie
d = Path(a.data); pv = pd.read_csv(d / 'plantvillage/meta.csv', index_col='idx'); ro = pd.read_csv(d / 'rocole/meta.csv', index_col='idx')
pv_img, ro_img = np.load(d / 'plantvillage/images.npy', mmap_mode='r'), np.load(d / 'rocole/images.npy', mmap_mode='r')

def b64(arr):
    buf = io.BytesIO(); Image.fromarray(np.array(arr)).save(buf, 'JPEG', quality=92); return base64.b64encode(buf.getvalue()).decode()

def call(instances):
    req = urllib.request.Request(f'{a.url.rstrip("/")}/v1/models/{a.model}:predict', data=json.dumps({'instances': instances}).encode(), headers={'Content-Type': 'application/json', **HEADERS})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r: out = json.loads(r.read())
    except urllib.error.HTTPError as e: sys.exit(f'the service answered {e.code}: {e.read().decode()[:300]}')
    except urllib.error.URLError as e: sys.exit(f'cannot reach {a.url}: {e.reason}')
    return out, (time.time() - t) * 1000

try:
    with urllib.request.urlopen(urllib.request.Request(f'{a.url.rstrip("/")}/v1/models/{a.model}', headers=HEADERS), timeout=15) as r: print('service is ready:', json.loads(r.read()))
except Exception as e: sys.exit(f'service not ready: {e}')

test = pv.index[pv['split'] == 'test']
lab_sick = [i for i in test if not pv.loc[i, 'healthy']][0]; lab_healthy = [i for i in test if pv.loc[i, 'healthy']][0]
dev = ro[ro['half'] == 'dev']; f_healthy = dev.index[~dev['sick']][:2].tolist(); f_sick = dev.index[dev['sick']][:2].tolist()
cases = [(f'lab leaf: {pv.loc[lab_sick, "cls"]}', pv_img[lab_sick]), (f'lab leaf: {pv.loc[lab_healthy, "cls"]}', pv_img[lab_healthy])] \
      + [(f'coffee, healthy (photo {i})', ro_img[i]) for i in f_healthy] + [(f'coffee, {ro.loc[i, "cls"]} (photo {i})', ro_img[i]) for i in f_sick] \
      + [('noise (not a leaf)', np.random.RandomState(0).randint(0, 255, (224, 224, 3), dtype=np.uint8))]
inst = [{'image_b64': b64(x)} for _, x in cases] + [{'image_b64': 'this is not an image'}]
names = [n for n, _ in cases] + ['broken input']
out, ms = call(inst)
print(f'\n{len(inst)} photos in one call: {ms:.0f} ms (the service says {out["milliseconds"]} ms inside) | model {out["model_version"]}\n')
print(f'{"photo":44s} {"p_sick":>6s}  {"triage":14s} {"decision":38s} top condition')
for n, p in zip(names, out['predictions']):
    if 'error' in p: print(f'{n:44s} {"-":>6s}  {p["error"]}'); continue
    print(f'{n:44s} {p["p_sick"]:6.2f}  {p["triage"]:14s} {p["decision"]:38s} {p["top3"][0]["condition"]} ({p["top3"][0]["p"]:.2f})')
print(f'\ncut-off used: {out["predictions"][0]["cutoff"]:.2f}   note: {out["note"]}')
t = [call([inst[2]])[1] for _ in range(a.repeat)]
print(f'one photo per call, {a.repeat} calls: median {np.median(t):.0f} ms, slowest {max(t):.0f} ms')
