# common/: shared helpers

`cropguard_lib.py` is imported by the scripts in `exercise3/` (training), `exercise4/` (pipeline steps) and `exercise5/` (the service); each adds this folder to its import path, so nothing is installed.
It holds what every exercise needs and Exercise 1 introduced: loading the packed dataset (and checking that the splits are clean), building ConvNeXt-Tiny, the "how sick?" score (1 minus the probability of all healthy classes),
the cut-off tuned on the dev half, evaluation (the locked half only when asked), loading a saved model folder (`model.pt` + `metadata.json`), and the photo preprocessing the model was trained with.

Where the exercises run: every pod mounts `data` at `/data` (dataset, models) and `home` at `/home/jovyan`, where the repository is cloned: `/home/jovyan/saudihpc-tutorials/kubeflow-tutorial/`.
