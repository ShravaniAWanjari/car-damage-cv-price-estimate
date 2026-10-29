# Fine-tune image for scripts/finetune_rfdetr.py.   Build from the repo root:
#   docker build -f deploy/train.Dockerfile -t REGISTRY/cardamage-train:TAG \
#       [--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128] .
# NOT built or run by the author (no Docker on the dev machine) - first build is the first test.
FROM python:3.11-slim-bookworm

# Optional: pin a CUDA build of torch matching your node driver (empty = default PyPI wheel).
ARG TORCH_INDEX_URL=""
# Roboflow COCO-pretrained starting weights are baked in so the Job needs no internet at run time.
ARG PRETRAIN_SIZES="medium large"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/opt/trainer-home \
    XDG_CACHE_HOME=/tmp/cache \
    TORCH_HOME=/tmp/torch \
    MPLCONFIGDIR=/tmp/mpl

RUN apt-get update \
 && apt-get install -y --no-install-recommends tini libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-job.txt .
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
      pip install torch==2.14.1 torchvision==0.29.1 --index-url "$TORCH_INDEX_URL"; \
    fi \
 && pip install -r requirements-job.txt

# Pre-download pretrained checkpoints into $HOME/.roboflow/models (rfdetr MD5-checks them).
RUN mkdir -p "$HOME" && python -c "import os, rfdetr; \
m={'nano':'RFDETRSegNano','small':'RFDETRSegSmall','medium':'RFDETRSegMedium','large':'RFDETRSegLarge'}; \
[getattr(rfdetr, m[s])(device='cpu') for s in os.environ['PRETRAIN_SIZES'].split()]" \
 && ls -la "$HOME/.roboflow/models"

COPY cardamage ./cardamage
COPY scripts ./scripts
COPY models.yaml ./models.yaml
COPY deploy/entrypoint_train.sh ./deploy/entrypoint_train.sh

# Run as an unprivileged user; the cluster's securityContext may override the uid, so keep things world-readable.
RUN chmod +x deploy/entrypoint_train.sh \
 && chmod -R a+rX /app "$HOME" \
 && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin trainer
USER 10001:10001

ENTRYPOINT ["tini", "--", "/app/deploy/entrypoint_train.sh"]
