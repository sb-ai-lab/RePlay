ARG python=3.10
FROM python:${python}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

COPY --from=eclipse-temurin:11 /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
RUN update-alternatives --install /usr/bin/java java /opt/java/openjdk/bin/java 1

RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils \
    && apt-get install libgomp1 build-essential pandoc -y \
    && apt-get install git -y --no-install-recommends

WORKDIR /root

RUN uv pip install --system --no-cache --upgrade pip wheel
