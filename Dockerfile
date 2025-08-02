FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install --no-install-recommends -y git ninja-build nvtop libegl1 libgl1 libgomp1 libglib2.0-0

ENV TORCH_CUDA_ARCH_LIST="5.0 6.0 7.0 7.5 8.0 8.6 9.0+PTX" FORCE_CUDA=1

ADD requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
