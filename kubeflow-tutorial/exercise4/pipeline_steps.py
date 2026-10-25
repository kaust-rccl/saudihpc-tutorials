"""Exercise 4: the steps of the CropGuard pipeline, as one script with a subcommand per step. (Training is train_ddp.py.)

Every step finds its files from three things: the models folder, a name prefix and a run id. The run is called <prefix>-<run id>; its model is
<models>/<run name>/ and its records (data report, evaluation, gate verdict) are in <models>/_pipeline/<run name>/.
To run a step by hand, name the run directly with --run-name (e.g. --run-name full1, a model folder you already have).

    python exercise4/pipeline_steps.py check-data --data /data/cropguard --models-dir /data/models/cropguard --run-name myrun
    python exercise4/pipeline_steps.py evaluate   --data /data/cropguard --models-dir /data/models/cropguard --run-name myrun      # also scores the current `production` model
    python exercise4/pipeline_steps.py gate       --models-dir /data/models/cropguard --run-name myrun [--min-lab-acc 0.95 --min-locked-auc 0.80 --max-regression 0.03]
    python exercise4/pipeline_steps.py publish    --models-dir /data/models/cropguard --run-name myrun

Kubeflow Pipelines runs the same commands in order, each in its own pod, and stops the chain if a step fails.
The FINAL EXAM happens in `evaluate`: it is the only place that scores the locked half of the coffee photos, once per model.
"""
import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))


def paths(a):
    run = a.run_name or f'{a.name_prefix}-{a.run_id}'; models = Path(a.models_dir)
    return run, models, models / run, models / '_pipeline' / run                       # run name, models folder, this run's model, this run's records


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + '.partial')
    tmp.write_text(json.dumps(obj, indent=1, default=float)); tmp.replace(p)


def check_data(a):
    import cropguard_lib as cg
    run, models, cand, recs = paths(a)
    rep = cg.Data(a.data, mmap=True).check()                                  # raises if photos are missing or the splits leak
    write_json(recs / 'data_report.json', rep); print(json.dumps(rep, indent=1)); print('data_check=ok')


def evaluate(a):
    import cropguard_lib as cg
    import torch
    run, models, cand, recs = paths(a)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = cg.Data(a.data)
    def score(path):
        model, meta = cg.load_saved_model(path, device)
        s = cg.evaluate_model(model, data, device, include_locked=True, cutoff=meta['cutoff'])          # the cut-off was tuned on the dev half when the model was trained
        return {'name': meta['name'], 'path': str(path), 'scores': s, 'training': {k: v for k, v in meta.get('training', {}).items() if k != 'history'}}
    t0 = time.time(); rep = {'candidate': score(cand), 'champion': None, 'device': str(device)}
    prod = models / 'production'
    if (prod / 'metadata.json').exists(): rep['champion'] = score(prod.resolve())
    rep['seconds'] = round(time.time() - t0, 1); write_json(recs / 'evaluation.json', rep)
    for who in ('candidate', 'champion'):
        if rep[who]: s = rep[who]['scores']; print(f'{who:9s} {rep[who]["name"]}: lab test acc {s["test_acc"]:.3f} | locked ranking score {s["locked_auc"]:.3f} | locked accuracy at its cut-off {s["locked_bal_acc_at_cutoff"]:.3f}')
    if not rep['champion']: print('no champion yet (first model): only the absolute floors apply')
    print(f'locked_auc={rep["candidate"]["scores"]["locked_auc"]:.4f}'); print(f'test_acc={rep["candidate"]["scores"]["test_acc"]:.4f}')


def gate(a):
    run, models, cand, recs = paths(a)
    rep = json.loads((recs / 'evaluation.json').read_text()); c = rep['candidate']['scores']; ch = rep['champion']['scores'] if rep['champion'] else None
    reasons, checks = [], []
    def rule(name, ok, detail):
        checks.append({'rule': name, 'passed': bool(ok), 'detail': detail})
        if not ok: reasons.append(f'{name}: {detail}')
    rule('lab accuracy floor', c['test_acc'] >= a.min_lab_acc, f'lab test accuracy {c["test_acc"]:.3f} (needs >= {a.min_lab_acc})')
    rule('real-world floor', c['locked_auc'] >= a.min_locked_auc, f'locked ranking score {c["locked_auc"]:.3f} (needs >= {a.min_locked_auc})')
    if ch: rule('no regression vs the current model', c['locked_auc'] >= ch['locked_auc'] - a.max_regression,
                f'candidate {c["locked_auc"]:.3f} vs current {ch["locked_auc"]:.3f} (may be at most {a.max_regression} lower: that is about the run-to-run noise)')
    decision = 'passed' if not reasons else 'blocked'
    out = {'decision': decision, 'reasons': reasons, 'checks': checks, 'candidate': rep['candidate']['name'], 'champion': rep['champion']['name'] if rep['champion'] else None,
           'thresholds': {'min_lab_acc': a.min_lab_acc, 'min_locked_auc': a.min_locked_auc, 'max_regression': a.max_regression}, 'time': datetime.now().isoformat(timespec='seconds')}
    write_json(recs / 'gate.json', out)
    if a.decision: p = Path(a.decision); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(decision)       # the pipeline reads this file to choose its next branch
    for k in checks: print(('PASS ' if k['passed'] else 'FAIL ') + k['rule'] + ': ' + k['detail'])
    print(f'decision={decision}')


def publish(a):
    run, models, cand, recs = paths(a); models = models.resolve(); cand = cand.resolve(); prod = models / 'production'
    gate_info = json.loads((recs / 'gate.json').read_text()) if (recs / 'gate.json').exists() else None
    if not gate_info or gate_info.get('decision') != 'passed': raise SystemExit('refusing to publish: there is no passing gate verdict for this run')
    if prod.exists() and not prod.is_symlink(): raise SystemExit(f'{prod} exists and is not a link: refusing to replace it')
    previous = os.readlink(prod) if prod.is_symlink() else None
    tmp = models / '.production.new'
    if tmp.is_symlink() or tmp.exists(): tmp.unlink()
    os.symlink(cand.name, tmp); os.replace(tmp, prod)                          # atomic: readers see the old model or the new one, never half
    with open(models / 'history.jsonl', 'a') as f: f.write(json.dumps({'time': datetime.now().isoformat(timespec='seconds'), 'published': cand.name, 'previous': previous, 'gate': gate_info}, default=float) + '\n')
    print(f'production -> {cand.name}' + (f' (was {previous}; to roll back: ln -sfn {previous} {prod})' if previous else ' (first model)')); print('published=yes')


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest='cmd', required=True)
def step(name, fn, data=False):
    p = sub.add_parser(name); p.add_argument('--models-dir', required=True); p.add_argument('--name-prefix', default='cropguard'); p.add_argument('--run-id', default=None); p.add_argument('--run-name', default=None)
    if data: p.add_argument('--data', required=True)
    p.set_defaults(fn=fn); return p
step('check-data', check_data, data=True); step('evaluate', evaluate, data=True)
p = step('gate', gate); p.add_argument('--decision', default=None, help='file to write the verdict (passed/blocked) into: the pipeline reads it')
p.add_argument('--min-lab-acc', type=float, default=0.95); p.add_argument('--min-locked-auc', type=float, default=0.80); p.add_argument('--max-regression', type=float, default=0.03)
step('publish', publish)
a = ap.parse_args()
if not (a.run_name or a.run_id): ap.error('give --run-name (by hand) or --run-id (the pipeline)')
a.fn(a)
