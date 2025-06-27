#!/bin/bash

docker cp pyspark_clasificacion_v2.py spark:pyspark_clasificacion_v2.py
docker exec spark spark-submit --master spark://spark:7077 --total-executor-cores 6 --executor-memory 1g /pyspark_clasificacion_v2.py
docker cp spark:/opt/bitnami/spark/resultados_prac3_cc.txt resultados_prac3_cc.txt
docker cp spark:/opt/bitnami/spark/resultados_prac3_cc_totales.txt resultados_prac3_cc_totales.txt
chown lassy:lassy resultados_prac3_cc.txt
chown lassy:lassy resultados_prac3_cc_totales.txt
