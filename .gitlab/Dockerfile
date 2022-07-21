FROM openjdk:11-bullseye

RUN apt-get update && apt-get install -y python3.9 python3.9-venv python3.9-dev build-essential
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py
RUN ln -s /usr/bin/pip3 /usr/bin/pip && \
    ln -s /usr/bin/python3.9 /usr/bin/python

RUN adduser --disabled-password --gecos "" user
USER user
WORKDIR /home/user

ENV PATH="/home/user/.local/bin:${PATH}"

RUN pip install --upgrade pip wheel poetry