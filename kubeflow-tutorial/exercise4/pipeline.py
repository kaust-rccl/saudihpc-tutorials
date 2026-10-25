"""Exercise 4: the CropGuard pipeline.  check data -> train -> evaluate (the final exam) -> gate -> publish (only if the gate passes).

    python exercise4/pipeline.py --out cropguard_pipeline.yaml               # compile (needs only `kfp`, which the notebook image has)
    python exercise4/run_pipeline.py                                         # compile and run it

Every step is a "container component": Kubeflow starts a pod with the notebook image and runs one of our scripts from the data volume
(exercise3/train_ddp.py, exercise4/pipeline_steps.py) from the repository clone in the home volume. Nothing needs installing, and each step can also be run by hand in a terminal.

What `kfp-kubernetes` does (mount the data and home volumes, put the Kueue label on the GPU pods) is written into the compiled file here with plain PyYAML
(`add_platform_spec`), so it also works in an image without that package. Checked identical to the package's output (image mshaikh/kubeflow-demo:kfp-torch-v1110 has it).
"""
import argparse, sys
import yaml
from kfp import compiler, dsl

DEFAULT_IMAGE = 'registry.tutorial.internal/ghcr/kubeflow/kubeflow/notebook-servers/jupyter-pytorch-cuda-full:v1.11.0'


REPO_DIR = '/home/jovyan/saudihpc-tutorials/kubeflow-tutorial'


def build(image=DEFAULT_IMAGE, repo_dir=REPO_DIR):
    steps, train_py = f'{repo_dir}/exercise4/pipeline_steps.py', f'{repo_dir}/exercise3/train_ddp.py'
    ident = lambda models, prefix, run_id: ['--models-dir', models, '--name-prefix', prefix, '--run-id', run_id]       # the run is called <prefix>-<this pipeline run's id>
    # The run id reaches every step as an ordinary input: the backend fills in `dsl.PIPELINE_JOB_ID_PLACEHOLDER` for task inputs, but not inside container arguments.

    @dsl.container_component
    def check_data(data_dir: str, models_dir: str, name_prefix: str, run_id: str):
        """Is the dataset all there, and are the leaf/plant splits still clean?"""
        return dsl.ContainerSpec(image=image, command=['python', steps, 'check-data'], args=['--data', data_dir] + ident(models_dir, name_prefix, run_id))

    @dsl.container_component
    def train(data_dir: str, models_dir: str, name_prefix: str, run_id: str, lr: float, epochs: int, seed: int):
        """Fine-tune the model on one GPU and save it as a new candidate (exactly what Exercise 3 did)."""
        return dsl.ContainerSpec(image=image, command=['python', train_py], args=['--data', data_dir, '--out', models_dir, '--name-prefix', name_prefix, '--run-id', run_id,
                                                                                '--lr', lr, '--epochs', epochs, '--seed', seed, '--workers', '0'])

    @dsl.container_component
    def evaluate(data_dir: str, models_dir: str, name_prefix: str, run_id: str):
        """The final exam: score the candidate (and the model currently in production) on the locked coffee photos."""
        return dsl.ContainerSpec(image=image, command=['python', steps, 'evaluate'], args=['--data', data_dir] + ident(models_dir, name_prefix, run_id))

    @dsl.container_component
    def gate(models_dir: str, name_prefix: str, run_id: str, min_lab_acc: float, min_locked_auc: float, max_regression: float, decision: dsl.OutputPath(str)):
        """Pass or block: floors for lab accuracy and real-world ranking, and no regression against the current model."""
        return dsl.ContainerSpec(image=image, command=['python', steps, 'gate'], args=ident(models_dir, name_prefix, run_id) + [
            '--decision', decision, '--min-lab-acc', min_lab_acc, '--min-locked-auc', min_locked_auc, '--max-regression', max_regression])

    @dsl.container_component
    def publish(models_dir: str, name_prefix: str, run_id: str):
        """Point `production` at the new model (atomic; the old one stays on disk, so rolling back is one command)."""
        return dsl.ContainerSpec(image=image, command=['python', steps, 'publish'], args=ident(models_dir, name_prefix, run_id))

    @dsl.pipeline(name='cropguard-train-evaluate-gate-publish', description='Check the data, train a candidate, take the final exam, and publish it only if it passes the gate.')
    def cropguard(data_dir: str = '/data/cropguard', models_dir: str = '/data/models/cropguard', name_prefix: str = 'cropguard',
                  lr: float = 0.0003, epochs: int = 2, seed: int = 42, min_lab_acc: float = 0.95, min_locked_auc: float = 0.80, max_regression: float = 0.03):
        rid = dsl.PIPELINE_JOB_ID_PLACEHOLDER
        t1 = check_data(data_dir=data_dir, models_dir=models_dir, name_prefix=name_prefix, run_id=rid)
        t2 = train(data_dir=data_dir, models_dir=models_dir, name_prefix=name_prefix, run_id=rid, lr=lr, epochs=epochs, seed=seed).after(t1); t2.set_accelerator_type('nvidia.com/gpu').set_accelerator_limit(1)
        t3 = evaluate(data_dir=data_dir, models_dir=models_dir, name_prefix=name_prefix, run_id=rid).after(t2); t3.set_accelerator_type('nvidia.com/gpu').set_accelerator_limit(1)
        t4 = gate(models_dir=models_dir, name_prefix=name_prefix, run_id=rid, min_lab_acc=min_lab_acc, min_locked_auc=min_locked_auc, max_regression=max_regression).after(t3)
        with dsl.If(t4.outputs['decision'] == 'passed', name='gate-passed'):
            publish(models_dir=models_dir, name_prefix=name_prefix, run_id=rid)

    return cropguard


def add_platform_spec(path, data_pvc='data', home_pvc='home', queue='batch', gpu_steps=('train', 'evaluate')):
    """Append the Kubernetes part of the pipeline (volumes, pod labels) to the compiled file, the way the `kfp-kubernetes` package would."""
    docs = list(yaml.safe_load_all(open(path))); ir = docs[0]
    def mount(pvc, where): return {'constant': pvc, 'mountPath': where, 'pvcNameParameter': {'runtimeValue': {'constant': pvc}}}
    execs = {}
    for name in ir['deploymentSpec']['executors']:                              # exec-check-data, exec-train, ...
        spec = {}
        if queue and name.removeprefix('exec-') in gpu_steps: spec['podMetadata'] = {'labels': {'kueue.x-k8s.io/queue-name': queue}}
        spec['pvcMount'] = [mount(data_pvc, '/data')] + ([mount(home_pvc, '/home/jovyan')] if home_pvc else [])
        execs[name] = spec
    platform = {'platforms': {'kubernetes': {'deploymentSpec': {'executors': execs}}}}
    with open(path, 'w') as f: yaml.safe_dump_all([ir, platform], f, sort_keys=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='cropguard_pipeline.yaml'); ap.add_argument('--image', default=DEFAULT_IMAGE); ap.add_argument('--repo-dir', default=REPO_DIR)
    ap.add_argument('--data-pvc', default='data'); ap.add_argument('--home-pvc', default='home'); ap.add_argument('--queue', default='batch', help="Kueue queue for the GPU steps ('' = no label)")
    a = ap.parse_args()
    compiler.Compiler().compile(build(a.image, a.repo_dir), a.out); add_platform_spec(a.out, a.data_pvc, a.home_pvc, a.queue)
    print(f'compiled {a.out}', file=sys.stderr)
