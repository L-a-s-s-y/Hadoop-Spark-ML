from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time

def entrenar_y_evaluar(nombre, classifier_stage, param_grid, usar_minmax=False, num_folds=5):
    
    stages = [label_indexer, assembler]
    if usar_minmax:
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
        stages.append(scaler)
    stages.append(classifier_stage)
    pipeline = Pipeline(stages=stages)
    
    builder = ParamGridBuilder()
    for param_name, valores in param_grid.items():
        param_obj = classifier_stage.getParam(param_name)
        builder = builder.addGrid(param_obj, valores)
    grid = builder.build()

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=num_folds
    )

    start_train = time.time()
    cvModel = cv.fit(train)
    end_train = time.time()
    train_elapsed = end_train - start_train
    train_times.append((nombre, train_elapsed))

    estimatorParamMaps = cv.getEstimatorParamMaps()
    avgMetrics = cvModel.avgMetrics
    for param_map, metric in zip(estimatorParamMaps, avgMetrics):
        params = {param.name: param_map[param] for param in param_map}
        resultados.append((nombre, params, metric))

    best_models.append((nombre, cvModel.bestModel))

if __name__ == "__main__":

    spark = SparkSession.builder.appName("Prac3_CC").getOrCreate()

    df = spark.read.csv(
        "hdfs://namenode:8020/user/spark/small_celestial.csv",
        sep=";", header=True, inferSchema=True
    ).dropna()

    df.printSchema()
    df.show(5)

    df.groupBy("type").count().orderBy("count", ascending=False).show()

    label_indexer = StringIndexer(inputCol="type", outputCol="indexedLabel")
    feature_cols = [c for c in df.columns if c != "type"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy"
    )
    resultados = []
    best_models = []
    train_times = []

    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")
    entrenar_y_evaluar(
        "Decision Tree",
        dt,
        {"maxDepth": [3, 5, 7, 10]},
        usar_minmax=False,
        num_folds=5
    )

    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features")
    entrenar_y_evaluar(
        "Random Forest",
        rf,
        {"numTrees": [3, 5, 7, 10]},
        usar_minmax=False,
        num_folds=5
    )

    lr = LogisticRegression(labelCol="indexedLabel", featuresCol="features")
    entrenar_y_evaluar(
        "Logistic Regression",
        lr,
        {"regParam": [0.01, 0.05, 0.1, 0.3]},
        usar_minmax=False,
        num_folds=5
    )

    print("\nResumen de resultados (parámetros y accuracy CV):")
    for modelo, params, acc in resultados:
        print(f"{modelo:20s} {params} -> CV Accuracy: {acc:.4f}")

    ruta_salida = "resultados_prac3_cc.txt"
    with open(ruta_salida, "w") as f:
        f.write("Resumen de resultados (parámetros y CV Accuracy):\n")
        for modelo, params, acc in resultados:
            f.write(f"{modelo:20s} {params} -> CV Accuracy: {acc:.4f}\n")
            
    print(f"\nResultados guardados en: {ruta_salida}")

    print("\n--- Evaluación en conjunto de test (Mejores Modelos) ---")
    ruta_test = "resultados_prac3_cc_totales.txt"
    with open(ruta_test, "w") as f:
        f.write("Evaluación en conjunto de test (Mejores Modelos):\n")
        for nombre, model in best_models:
            start_test = time.time()
            pred = model.transform(test)
            end_test = time.time()
            test_elapsed = end_test - start_test

            metrics = {}
            for metric_name in ["accuracy", "weightedPrecision", "weightedRecall", "f1"]:
                eval_temp = MulticlassClassificationEvaluator(
                    labelCol="indexedLabel", predictionCol="prediction", metricName=metric_name
                )
                metrics[metric_name] = eval_temp.evaluate(pred)

            train_time = next(t for n, t in train_times if n == nombre)
            metrics['trainTime'] = train_time
            metrics['testTime'] = test_elapsed

            linea = (f"{nombre:20s} " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
            print(linea)
            f.write(linea + "\n")
    print(f"Resultados de test guardados en: {ruta_test}")

    spark.stop()
