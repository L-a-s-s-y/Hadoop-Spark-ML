#!/bin/bash

systemctl start docker
docker compose up -d
sleep 2

# A veces el contenedor no está listo llegados este momento y dará
# un error. Si eso sucede simplemente volver a lanzar el script

#Namenode
docker container ls -a
docker cp dataset/small_celestial.csv namenode:/data/small_celestial.csv
docker exec namenode hdfs dfs -mkdir /user
docker exec namenode hdfs dfs -mkdir /user/spark
docker exec namenode hdfs dfs -chown spark:spark /user/spark
docker exec namenode hdfs dfs -ls /user/spark
docker exec namenode hdfs dfs -put /data/small_celestial.csv /user/spark/

#Spark
docker cp pyspark_clasificacion_v2.py spark:pyspark_clasificacion_v2.py
