ARG python
FROM python:${python}-slim-bookworm AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    libgomp1 \
    build-essential \
    pandoc \
    git \
    openjdk-17-jdk-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root

RUN pip install --no-cache-dir --upgrade pip wheel poetry==2.2.1 poetry-dynamic-versioning
