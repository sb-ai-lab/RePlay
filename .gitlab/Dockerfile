ARG python
FROM python:${python}-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

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
