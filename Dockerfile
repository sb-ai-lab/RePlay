FROM python:3.11-slim as builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils \
    && apt-get install libgomp1 build-essential pandoc -y \
    && apt-get install git -y --no-install-recommends

COPY --from=openjdk:11-jre-slim /usr/local/openjdk-11 /usr/local/openjdk-11
ENV JAVA_HOME /usr/local/openjdk-11
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-11/bin/java 1

WORKDIR /root

RUN pip install --no-cache-dir --upgrade pip wheel poetry==1.5.1 poetry-dynamic-versioning \
    && python -m poetry config virtualenvs.create false
COPY . RePlay-Accelerated/
RUN cd RePlay-Accelerated && ./poetry_wrapper.sh install --all-extras
CMD ["/bin/bash"]