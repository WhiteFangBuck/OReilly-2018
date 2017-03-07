package com.cloudera.workshop.exercises

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

/**
  * Created by jayant on 3/6/17.
  */
object BikeSharingAnalysis {

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder
      .appName("ChurnExample")
      .master("local")
      .getOrCreate()

    // read in the data into a DataFrame
    val ds = spark.read.option("inferSchema", "true").option("header", "true").csv("data/bike_sharing_sample_dataset.csv")
    ds.printSchema()

    // convert date
    val func = unix_timestamp(col("datetime"), "dd/MM/yyyy HH:mm").cast("timestamp")
    var newds = ds.withColumn("datetime_cast", func).drop("datetime").withColumnRenamed("datetime_cast", "datetime")

    // extract year, month, dayof month, hour as a new column
    newds = newds.select(col("*"), year(newds.col("datetime")).alias("datetime_year"))
    newds = newds.select(col("*"), month(newds.col("datetime")).alias("datetime_month"))
    newds = newds.select(col("*"), dayofmonth(newds.col("datetime")).alias("datetime_dayofmonth"))
    newds = newds.select(col("*"), hour(newds.col("datetime")).alias("datetime_hour"))

    // vector assembler
    val assembler = new VectorAssembler()
      .setInputCols(Array("season", "holiday", "workingday", "weather",
        "humidity", "windspeed", "temp", "atemp",
        "datetime_year", "datetime_month", "datetime_dayofmonth", "datetime_hour"))
      .setOutputCol("features")

    newds = assembler.transform(newds)
    newds.printSchema()

    // vector assembler
    val vectorindexer = new VectorIndexer().setInputCol("features").setMaxCategories(31).setOutputCol("features_vector_index")

    val indexerModel = vectorindexer.fit(newds)
    newds = indexerModel.transform(newds)

    // split the data for training and test
    val Array(trainingData, testData) = newds.randomSplit(Array(0.7, 0.3), 1000)

    // show training data
    trainingData.printSchema()
    trainingData.show(10)

    // Train a GBT model.
    val gbt = new GBTRegressor()
      .setLabelCol("count")
      .setFeaturesCol("features_vector_index")
      .setMaxIter(10)

    // train model
    val gbtModel = gbt.fit(trainingData)

    // Make predictions on test data
    val predictions = gbtModel.transform(testData)
    predictions.printSchema()

    // Select (prediction, true label) and compute test error.
    predictions.select("prediction", "count", "features_vector_index").show(5)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("count")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    spark.stop()

  }

}
