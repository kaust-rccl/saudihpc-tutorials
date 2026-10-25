#!/bin/bash
# Build the image for amd64 and arm64 and push it to Docker Hub (needs `docker login` as the owner of the repository).
#   bash exercise4/image/build_and_push.sh            # from kubeflow-tutorial/
set -euo pipefail
cd "$(dirname "$0")"; TAG=${TAG:-mshaikh/kubeflow-demo:kfp-torch-v1110}; V=${KFP_KUBERNETES_VERSION:-2.16.1}
rm -rf wheel site-packages; mkdir wheel site-packages
python3 -m pip download "kfp-kubernetes==$V" --no-deps --only-binary=:all: -d wheel -q
python3 -c "import zipfile,glob; zipfile.ZipFile(glob.glob('wheel/*.whl')[0]).extractall('site-packages')"
BUILDER=${BUILDER:-multiarch}     # a docker-container builder: the default docker driver cannot build for several platforms
docker buildx build --builder "$BUILDER" --platform linux/amd64,linux/arm64 -t "$TAG" --push .
docker buildx imagetools inspect "$TAG" | head -12
