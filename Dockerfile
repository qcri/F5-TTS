# syntax=docker/dockerfile:1.7
# ---------------------------
# F5-TTS Base Image
# ---------------------------
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS builder

USER root
ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/SWivid/F5-TTS"

ENV PIP_CACHE_DIR=/root/.cache/pip

# System dependencies
RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       wget curl git openssl libssl-dev unzip build-essential aria2 \
       sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
       librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/F5-TTS

COPY . /workspace/F5-TTS

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -e . --cache-dir=$PIP_CACHE_DIR

# ---------------------------
# Runtime Image
# ---------------------------
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace/F5-TTS

# Copy installed environment + repo from builder
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /workspace/F5-TTS /workspace/F5-TTS

ENV PATH="/opt/conda/bin:$PATH"
ENV SHELL=/bin/bash
