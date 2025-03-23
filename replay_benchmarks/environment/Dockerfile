# Start with the CUDA base image
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
ENV PYENV_ROOT /opt/pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:/opt/miniconda3/bin:$PATH

# Remove any third-party apt sources to avoid issues with expiring keys
RUN rm -f /etc/apt/sources.list.d/*.list

# Install essential system dependencies
RUN apt-get update && apt-get install -y \
    curl wget mc unzip htop vim sudo adduser \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev default-jdk scala \
    openjdk-8-jdk ant ca-certificates-java \
    ffmpeg libsm6 libxext6 sox libsox-dev libsox-fmt-all tmux \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
ENV PATH $JAVA_HOME/bin:$PATH

# Set root password as empty
RUN passwd -d root

# Set sudoers for root and all users
RUN sed -i '/^.*$/d' /etc/sudoers && \
    echo 'root ALL=(ALL:ALL) ALL' >> /etc/sudoers && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Add developers group and users
RUN addgroup --gid 1000 developers && \
    for user in $(getent passwd | cut -d: -f1); do \
      if [[ $user == "da" ]] || [[ $(id -u $user) -ge 1000 ]] && [[ $(eval echo "~$user") == /home* ]]; then \
        addgroup --gid $(id -g $user) $user && \
        adduser --disabled-password --gecos "" --uid $(id -u $user) --gid $(id -g $user) --shell $(getent passwd $user | cut -d: -f7) --home $(eval echo "~$user") $user && \
        adduser $user sudo && adduser $user developers; \
      fi; \
    done

# Set up Miniconda for environment and package management
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda3 \
    && rm ~/miniconda.sh

# Update PATH to include Miniconda
ENV PATH="/opt/miniconda3/bin:$PATH"

# Install Mamba
RUN conda install -n base -c conda-forge mamba

# Copy and install environment using Mamba
COPY environment.yml /tmp/environment.yml
RUN mamba env create --file=/tmp/environment.yml \
    && conda clean -ya \
    && rm /tmp/environment.yml

# Install additional Python packages via pip
RUN conda run -n env pip install hydra-core implicit lightfm torch pytorch-lightning sqlalchemy hdfs redis msgpack_numpy msgpack lz4 nvitop

# Set up default Conda environment and activate it
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

# Download and setup Hadoop
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.1/hadoop-3.3.1.tar.gz && \
    tar -xzf hadoop-3.3.1.tar.gz -C /opt && \
    rm hadoop-3.3.1.tar.gz

# Set environment variables for Hadoop
ENV HADOOP_HOME=/opt/hadoop-3.3.1
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$LD_LIBRARY_PATH
ENV PATH=$HADOOP_HOME/bin:$PATH

# Download and setup Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz && \
    tar -xzf spark-3.4.1-bin-hadoop3.tgz -C /opt && \
    ln -s /opt/spark-3.4.1-bin-hadoop3 /opt/spark && \
    rm spark-3.4.1-bin-hadoop3.tgz

# Set environment variables for Spark
ENV SPARK_HOME /opt/spark
ENV PATH $SPARK_HOME/bin:$PATH