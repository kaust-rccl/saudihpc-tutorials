"""Exercise 5: write the KServe InferenceService that serves the model `production` points to. Defaults are for the T4 cluster (PVCs `data` -> /data and `home` -> /home/jovyan).

    python exercise5/make_inferenceservice.py --namespace <ns> --cluster aws.mlops101.io | kubectl apply -f -
    kubectl get inferenceservice cropguard -n <ns>            # wait for READY=True
    python exercise5/call_service.py --namespace <ns>                                    # from the notebook: inside the cluster, no login needed
    python exercise5/call_service.py --cluster aws.mlops101.io --namespace <ns> --token-file <token>   # from outside, through the cluster address

Two addresses reach the same service:
  * inside the cluster:  http://cropguard-predictor.<ns>.svc.cluster.local/v1/models/cropguard:predict
  * from outside:        https://<cluster-address>/serving/<ns>/cropguard/v1/models/cropguard:predict   (an Ingress on the cluster gateway; the gateway asks you to log in)

It runs exercise5/serve_cropguard.py from the repository clone in the home volume, in the notebook image (nothing to build or push), on CPU: one photo takes about 0.1 s,
so a GPU is not needed and no queue is involved. RawDeployment mode is a plain Deployment plus Service (no Knative), which can mount the volumes.
"""
import argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--name', default='cropguard'); ap.add_argument('--namespace', default=None)
ap.add_argument('--image', default='registry.tutorial.internal/ghcr/kubeflow/kubeflow/notebook-servers/jupyter-pytorch-cuda-full:v1.11.0')
ap.add_argument('--data-pvc', default='data'); ap.add_argument('--home-pvc', default='home'); ap.add_argument('--repo-dir', default='/home/jovyan/saudihpc-tutorials/kubeflow-tutorial')
ap.add_argument('--model-dir', default='/data/models/cropguard/production')
ap.add_argument('--mode', default='RawDeployment', choices=['RawDeployment', 'Serverless'], help='Serverless (Knative) cannot mount a volume unless the cluster enables it')
ap.add_argument('--cluster', default=None, help='the cluster address (e.g. aws.mlops101.io): only used to print the external URL in the header')
ap.add_argument('--no-route', action='store_true', help='do not add the Ingress for https://<cluster-address>/serving/<namespace>/<name> (in-cluster calls still work)')
ap.add_argument('--ingress-class', default='istio'); ap.add_argument('--path-template', default='/serving/{namespace}/{name}', help="must match the cluster's KServe pathTemplate")
ap.add_argument('--replicas', type=int, default=1); ap.add_argument('--cpu', default='2'); ap.add_argument('--memory', default='4Gi'); ap.add_argument('--out', default=None)
a = ap.parse_args()
ns = a.namespace
if ns is None:
    f = Path('/var/run/secrets/kubernetes.io/serviceaccount/namespace'); ns = f.read_text().strip() if f.exists() else '<your-namespace>'
base = a.path_template.format(namespace=ns, name=a.name)
cluster = a.cluster or '<cluster-address>'
header = f'# InferenceService "{a.name}"\n# inside the cluster: http://{a.name}-predictor.{ns}.svc.cluster.local/v1/models/{a.name}:predict\n'
if not a.no_route: header += f'# from outside:       https://{cluster}{base}/v1/models/{a.name}:predict   (the cluster gateway asks for a login)\n'
env_base = '' if a.no_route else f'\n      - {{name: BASE_PATH, value: {base}}}'
isvc = f'''apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {a.name}
  namespace: {ns}
  annotations:
    serving.kserve.io/deploymentMode: {a.mode}
    sidecar.istio.io/inject: "false"          # plain in-cluster HTTP; the Istio sidecar is not needed for this exercise
spec:
  predictor:
    minReplicas: {a.replicas}
    maxReplicas: {a.replicas}
    containers:
    - name: kserve-container                  # KServe expects this name for a custom container
      image: {a.image}
      imagePullPolicy: IfNotPresent
      command: [python, {a.repo_dir}/exercise5/serve_cropguard.py]
      env:
      - {{name: MODEL_DIR, value: {a.model_dir}}}
      - {{name: MODEL_NAME, value: {a.name}}}
      - {{name: PYTHONUNBUFFERED, value: "1"}}{env_base}
      ports:
      - {{containerPort: 8080, protocol: TCP}}
      resources:
        requests: {{cpu: "1", memory: 2Gi}}
        limits: {{cpu: "{a.cpu}", memory: {a.memory}}}
      readinessProbe:
        httpGet: {{path: /v1/models/{a.name}, port: 8080}}
        initialDelaySeconds: 5
        periodSeconds: 5
      volumeMounts:
      - {{name: data, mountPath: /data, readOnly: true}}
      - {{name: home, mountPath: /home/jovyan, readOnly: true}}
    volumes:
    - name: data
      persistentVolumeClaim: {{claimName: {a.data_pvc}}}
    - name: home
      persistentVolumeClaim: {{claimName: {a.home_pvc}}}
'''
route = f'''---
# The address from outside: any host name, this path prefix, on the cluster gateway. (A plain Ingress cannot rewrite paths, so the service accepts the prefix: BASE_PATH.)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {a.name}-path
  namespace: {ns}
spec:
  ingressClassName: {a.ingress_class}
  rules:
  - http:
      paths:
      - path: {base}
        pathType: Prefix
        backend:
          service:
            name: {a.name}-predictor
            port: {{number: 80}}
'''
text = header + isvc + ('' if a.no_route else route)
if a.out: Path(a.out).write_text(text); print(f'wrote {a.out}', file=sys.stderr)
else: sys.stdout.write(text)
