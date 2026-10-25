# The Kubeflow Jupyter PyTorch CUDA notebook image plus the `kfp-kubernetes` package (volume mounts, pod labels and the like for Kubeflow Pipelines steps).
# kfp-kubernetes is pure Python, so its files are copied in rather than installed: nothing runs inside the image during the build,
# which means the same Dockerfile builds linux/amd64 and linux/arm64 without emulation. Its dependencies (kfp 2.16.1, protobuf 6.x) are already in the base image.
# Rebuild after changing the version:  see build_and_push.sh
FROM ghcr.io/kubeflow/kubeflow/notebook-servers/jupyter-pytorch-cuda-full:v1.11.0
COPY --chmod=a+rX site-packages/ /opt/conda/lib/python3.12/site-packages/
LABEL org.opencontainers.image.description="kubeflow notebook-servers/jupyter-pytorch-cuda-full:v1.11.0 + kfp-kubernetes 2.16.1"
