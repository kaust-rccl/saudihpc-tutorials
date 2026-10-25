"""Exercise 5: the CropGuard prediction service, a KServe "custom container" predictor that speaks the KServe v1 protocol.
Standard library + PyTorch only (the notebook image has no `kserve` or web-framework package).

    POST /v1/models/cropguard:predict   {"instances": [{"image_b64": "<JPEG or PNG, base64>"}, ...]}
        ->  {"predictions": [{"p_sick": 0.81, "triage": "likely sick", "cutoff": 0.74, "decision": "diagnosis",
                              "top3": [{"condition": "Tomato___Late_blight", "p": 0.43}, ...], "model": "<run name>"}], ...}
    GET  /v1/models/cropguard           readiness: {"name": "cropguard", "ready": true, ...}

Settings (environment variables): MODEL_DIR (default /data/models/cropguard/production), MODEL_NAME (cropguard), PORT (8080), TORCH_THREADS (2), SERVE_DEVICE (cpu),
BASE_PATH (default empty): a prefix such as /serving/<namespace>/cropguard that the cluster's gateway leaves on the URL; it is stripped before routing, so the same service answers
at http://<service>/v1/models/... inside the cluster and at https://<cluster-address>/serving/<namespace>/cropguard/v1/models/... from outside.

`production` is a link that the pipeline's publish step moves. The service looks where it points on every request and reloads when it changes,
so publishing a model (or rolling back with `ln -sfn`) takes effect without redeploying anything.

Two numbers decide what the app tells the farmer, and both are saved next to the model (metadata.json), not hard-coded here:
  * `cutoff`: call a leaf "likely sick" if its sick score is at least this (tuned on the dev half when the model was trained: an inference-time setting);
  * `confidence_threshold`: if the model's most likely condition has less probability than this, the answer is "refer to extension officer".
"""
import base64, json, os, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import numpy as np, torch
import cropguard_lib as cg

NAME, PORT = os.environ.get('MODEL_NAME', 'cropguard'), int(os.environ.get('PORT', '8080'))
MODEL_DIR = os.environ.get('MODEL_DIR', '/data/models/cropguard/production')
DEVICE = torch.device(os.environ.get('SERVE_DEVICE', 'cpu')); torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '2')))
MAX_BODY = 25 * 1024 * 1024
BASE_PATH = os.environ.get('BASE_PATH', '').rstrip('/')
def route(raw):
    p = raw.split('?')[0]
    return (p[len(BASE_PATH):] if BASE_PATH and p.startswith(BASE_PATH) else p).rstrip('/')
NOTE = 'conditions are the 38 PlantVillage classes (14 crops); for any other crop, e.g. coffee, trust the sick/healthy triage only'

_lock, _cur = threading.Lock(), {'real': None, 'model': None, 'meta': None, 'loaded': None}


def current():
    """The model that `production` points to now, loaded once and reloaded whenever the link moves."""
    real = os.path.realpath(MODEL_DIR)
    if _cur['real'] != real:
        with _lock:
            if _cur['real'] != real:
                t = time.time(); model, meta = cg.load_saved_model(real, DEVICE)
                _cur.update(real=real, model=model, meta=meta, loaded=time.strftime('%Y-%m-%d %H:%M:%S'))
                print(f'loaded model "{meta["name"]}" from {real} in {time.time() - t:.1f}s (cutoff {meta["cutoff"]:.2f})', flush=True)
    return _cur['model'], _cur['meta']


def predict(instances):
    model, meta = current(); ok, photos, out = [], [], [None] * len(instances)
    for i, inst in enumerate(instances):
        try: photos.append(cg.photo_from_bytes(base64.b64decode(inst['image_b64'], validate=True))); ok.append(i)
        except Exception as e: out[i] = {'error': 'could not read the image (it must be a base64-encoded JPEG or PNG)'}
    if photos:
        P = cg.predict_proba(model, np.stack(photos), DEVICE); healthy = np.array(meta['healthy_class_ids']); s = cg.triage_score(P, healthy)
        for k, i in enumerate(ok):
            top = np.argsort(P[k])[::-1][:3]; conf = float(P[k].max())
            out[i] = {'p_sick': round(float(s[k]), 4), 'triage': 'likely sick' if s[k] >= meta['cutoff'] else 'likely healthy', 'cutoff': round(meta['cutoff'], 2),
                      'decision': 'diagnosis' if conf >= meta['confidence_threshold'] else 'refer to extension officer (low confidence)',
                      'top3': [{'condition': meta['classes'][j], 'p': round(float(P[k][j]), 4)} for j in top], 'model': meta['name']}
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = 'cropguard/1'
    def log_message(self, fmt, *args): pass                                          # we log our own, one line per prediction

    def send(self, code, obj):
        body = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = route(self.path)
        if path in ('', '/healthz', '/ready', '/live'): return self.send(200, {'status': 'alive'})
        if path == '/v1/models': return self.send(200, {'models': [NAME]})
        if path == f'/v1/models/{NAME}':
            try: _, meta = current(); return self.send(200, {'name': NAME, 'ready': True, 'model': meta['name'], 'cutoff': round(meta['cutoff'], 2), 'loaded': _cur['loaded']})
            except Exception as e: return self.send(503, {'name': NAME, 'ready': False, 'error': str(e)[:200]})
        self.send(404, {'error': 'not found'})

    def do_POST(self):
        path = route(self.path)
        if path != f'/v1/models/{NAME}:predict': return self.send(404 if not path.endswith(':explain') else 501, {'error': 'only :predict is implemented'})
        n = int(self.headers.get('Content-Length', 0))
        if n <= 0 or n > MAX_BODY: return self.send(413 if n > MAX_BODY else 400, {'error': 'empty or too large request'})
        t = time.time()
        try:
            inst = json.loads(self.rfile.read(n))['instances']
            if not isinstance(inst, list) or not inst: raise ValueError('"instances" must be a non-empty list')
            preds = predict(inst)
        except (KeyError, ValueError, json.JSONDecodeError) as e: return self.send(400, {'error': f'bad request: {e}'})
        except Exception as e: traceback.print_exc(); return self.send(500, {'error': f'prediction failed: {type(e).__name__}'})
        ms = (time.time() - t) * 1000; print(f'predict: {len(preds)} photo(s) in {ms:.0f} ms', flush=True)
        self.send(200, {'predictions': preds, 'model_name': NAME, 'model_version': _cur['meta']['name'], 'milliseconds': round(ms), 'note': NOTE})


if __name__ == '__main__':
    try: current()                                                                  # load before accepting traffic; if the model is missing the pod fails fast and says why
    except Exception as e: print(f'cannot load a model from {MODEL_DIR}: {e}', flush=True); raise
    print(f'serving "{NAME}" on port {PORT} (device {DEVICE})', flush=True)
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()
