ARG python
FROM python:${python}-slim as builder
RUN apt-get update \
    && apt-get install wget apt-utils libgomp1 build-essential pandoc git -y --no-install-recommends

USER root
WORKDIR /root

COPY --from=openjdk:11-jre-slim /usr/local/openjdk-11 /usr/local/openjdk-11
ENV JAVA_HOME /usr/local/openjdk-11
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk-11/bin/java 1

# Install Spark 3.4.3 with Hadoop 3
RUN wget https://dlcdn.apache.org/spark/spark-3.4.3/spark-3.4.3-bin-hadoop3.tgz -P /opt/
RUN tar -zxf /opt/spark-3.4.3-bin-hadoop3.tgz -C "/opt/"
RUN ln -s /opt/spark-3.4.3-bin-hadoop3 /opt/spark
RUN echo "export PYSPARK_PYTHON=/usr/local/bin/python3" >> /opt/spark/conf/spark-env.sh
RUN echo "export PYSPARK_DRIVER_PYTHON=python3" >> /opt/spark/conf/spark-env.sh
RUN echo "spark.hadoop.fs.s3a.aws.credentials.provider org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider" >> /opt/spark/conf/spark-defaults.conf

ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

RUN pip install --no-cache-dir --upgrade pip wheel poetry==1.5.1 poetry-dynamic-versioning lightfm==1.17 \
    && python -m poetry config virtualenvs.create false
COPY . replay/
RUN cd replay && ./poetry_wrapper.sh install --all-extras
RUN rm -rf replay/
