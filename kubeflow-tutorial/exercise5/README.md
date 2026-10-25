# Exercise 5: KServe, from a file on a disk to a service

**Status: written and tested on the dev cluster (KServe 0.18, RawDeployment), inside the cluster. The address from outside is NOT verified (see below). Not yet run on the T4 cluster.**

## The story
A model on a volume helps nobody. The farm-advisory app needs an **address it can send a photo to**. KServe turns "a model folder" into "a service": a Deployment plus a Service, health checks, and one standard request format.

The service you deploy reads whatever the pipeline last published as `production`, and answers three questions for every photo:
* **How sick does this look?** (`p_sick`, the sick score from Exercise 1) and, using the **cut-off** that was tuned for this model and saved next to it (`metadata.json`), **likely sick or likely healthy?**
* **Which condition?** (the three most likely of the 38 trained conditions)
* **Should the app answer at all?** If the model is not confident, the decision is "refer to extension officer" instead of a guess.

Three ideas to land:
1. **The cut-off is an inference-time setting**, not part of training. It travels with the model (`metadata.json`), so a new model brings its own: we saw the service move from cut-off 0.74 to 0.71 when a new model was published.
2. **`production` is just a pointer.** Publish a new model from the pipeline (or roll back with `ln -sfn <old-run> /data/models/cropguard/production`) and the live service switches within a request, with no redeploy and no restart (measured: 0 restarts).
3. **Honest limits.** On coffee photos the *triage* (sick or healthy) is the useful answer; the named conditions belong to the 14 PlantVillage crops, and the service says so (the "note" field, and the low-confidence decisions for coffee). This is a demonstration service: no authentication of its own, no monitoring, no gradual rollout.

## Layout
`data` at `/data` and `home` at `/home/jovyan` are mounted (read-only) in the service pod, which runs `exercise5/serve_cropguard.py` from the repository clone in the home volume, in the notebook image. Nothing is built or pushed.

| File | What |
|---|---|
| `serve_cropguard.py` | the service: standard library plus PyTorch, KServe v1 protocol (`POST /v1/models/cropguard:predict`) |
| `make_inferenceservice.py` | writes the InferenceService and the Ingress for the outside address |
| `call_service.py` | the "phone app": sends real dataset photos plus two awkward inputs, prints the answers and timings |
| `inferenceservice.yaml` | a sample manifest with the T4 defaults, to read |
| `../common/cropguard_lib.py` | shared helpers: model loading, preprocessing, scores |

## The service address
| From | Address |
|---|---|
| inside the cluster (a notebook, a pipeline step) | `http://cropguard-predictor.<namespace>.svc.cluster.local/v1/models/cropguard:predict` (no login) |
| outside, through the cluster address | `https://<cluster-address>/serving/<namespace>/cropguard/v1/models/cropguard:predict`, for example `https://aws.mlops101.io/serving/<namespace>/cropguard/v1/...` (the gateway asks you to log in) |

`make_inferenceservice.py` adds an Ingress for the outside path (`/serving/<namespace>/<name>`, any host name); the service strips that prefix itself (`BASE_PATH`) because a plain Ingress cannot rewrite paths. `--no-route` leaves the Ingress out.
Two things depend on your cluster and could not be checked on the dev cluster, which has no ingress class at all:
* **The Ingress class.** The default is `istio`. KServe also creates its own Ingress for the service (host-based, `<name>-<namespace>.<ingress domain>`): `kubectl get ingress,inferenceservice -n <ns>` shows its class and the URL it expects. If that Ingress works on your cluster, pass its class with `--ingress-class`; the URL KServe printed is a second way in.
* **The login.** The cluster gateway only accepts logged-in callers. `call_service.py --cluster ... --token-file F` sends `Authorization: Bearer <token>` (a token your login service accepts); `--cookie "name=value"` sends a browser session cookie instead (copy the cookie of a logged-in Kubeflow session from the browser's developer tools). From inside the cluster none of this is needed.

## Run on the T4 cluster
You need a model in `production`: run the Exercise 4 pipeline once (or `exercise4/pipeline_steps.py publish` by hand). From `/home/jovyan/saudihpc-tutorials/kubeflow-tutorial`:
```bash
ls -l /data/models/cropguard/production                                          # a link to a model folder
python exercise5/make_inferenceservice.py --cluster aws.mlops101.io | kubectl apply -f -
kubectl get inferenceservice cropguard -w                                        # wait for READY=True (about 15 seconds)
python exercise5/call_service.py                                                 # inside the cluster, from the notebook; no arguments needed
python exercise5/call_service.py --cluster aws.mlops101.io --token-file my_token # from outside, through the cluster address
python exercise5/call_service.py --cluster aws.mlops101.io --print-url           # just show the address it would call
```
Then try it live: publish a different model and call again, or roll back:
```bash
cat /data/models/cropguard/history.jsonl                                          # what was published, and what it replaced
ln -sfn <older-run-folder> /data/models/cropguard/production                      # roll back (the data volume is writable from the notebook)
python exercise5/call_service.py                                                 # the "service is ready" line shows the new model name
```
Tear down: `kubectl delete inferenceservice cropguard; kubectl delete ingress cropguard-path`.
Useful: `kubectl logs deploy/cropguard-predictor` (one line per prediction, and a line each time a model is loaded), `make_inferenceservice.py --help` (image, volumes, CPU, memory, `--mode Serverless`).

## What was tested here, and what was not
**Tested (dev cluster):** both manifests are accepted; the service became Ready in about 15 seconds; calls from a pod inside the cluster, with no arguments and also through the `/serving/<ns>/cropguard` prefix, returned correct, sensible answers
(lab leaves: confident diagnoses; coffee photos: triage right, decision "refer to extension officer"; noise and a broken image handled); about 0.18 s per photo on 2 CPUs;
switching `production` to another model and back, and publishing from the Exercise 4 pipeline, changed the served model with no restart (and brought the new model's own cut-off).
**Not tested, check first on the T4 cluster:**
1. **The outside address and the login** (above): the Ingress class, the path, the token or cookie.
2. **The in-cluster address.** `cropguard-predictor.<ns>.svc.cluster.local` is how KServe names the Service in RawDeployment mode on the dev cluster; `kubectl get svc -n <ns>` shows yours.
3. **Policies:** the Harbor image path is copied from your Katib file. The service asks for CPU and memory only (no GPU, no queue). If a policy requires something else, the admission message will say what.
4. **Raw versus Serverless:** RawDeployment is used because Knative (Serverless) refuses volume mounts unless the cluster enables them; on the dev cluster they are off.
5. **Volumes:** both are mounted read-only; with ReadWriteOnce volumes the pod must land on the node that already holds them.
6. **Sidecar:** `sidecar.istio.io/inject: "false"` is on the service so plain in-cluster HTTP works; remove it if your mesh requires the sidecar for traffic to reach the pod.
