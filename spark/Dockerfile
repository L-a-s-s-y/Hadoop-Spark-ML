FROM docker.io/bitnami/spark:3.3.0

USER root
RUN apt-get update && apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# (Optional) Ensure pyspark is installed if needed. In many cases, Spark's PySpark is already included,
# but if you need a specific version or extra Python packages, do so here:
# RUN pip3 install pyspark==3.3.0

RUN pip3 install numpy

USER 1001

