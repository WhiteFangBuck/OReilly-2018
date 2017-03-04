package com.cloudera.workshop

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
/**
 * A simple example demonstrating model selection using CrossValidator.
 * This example also demonstrates how Pipelines are Estimators.
 *
 */
object ProblemTwoCV {

  def main(args: Array[String]): Unit = {
    Logger.getRootLogger.setLevel(Level.OFF)
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession
      .builder
      .master("local[4]")
      .appName("ProblemTwoCV")
      .getOrCreate()

    /**
      * This is your training data
      */
    val training = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0),
      (4L, "b spark who", 1.0),
      (5L, "g d a y", 0.0),
      (6L, "spark fly", 1.0),
      (7L, "was mapreduce", 0.0),
      (8L, "e spark program", 1.0),
      (9L, "a e c l", 0.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 0.0)
    )).toDF("id", "text", "label")

    /**
      * Tokenize the \"text"\ column
      * Start of the pipeline
      */
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    /**
      * Use the HashingTF on Tokenizer output
      */
    val hashingTF = new HashingTF()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    /**
      * Initialize a logistic regression model
      */
    val lr = new LogisticRegression()
      .setMaxIter(10)

    /**
      * Initialize the pipeline using previous three nodes
      */
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    /**
      * Create the parameter builder using various number of features and different values for the regularizer
      */
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    /**
      * Initiate a CrossValidator
      * Treate the pipeline as an estimator and a BinaryClassificationEvaluator as testor.
      * NumberofFolds are 2+.
      */
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2) // Use 3+ in practice

    /**
      * Run cross validation
      */
    val cvModel = cv.fit(training)

    /**
      * Prepare the test documents
      */
    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "mapreduce spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    /**
      * Make predictions on the test documents
      */
    cvModel.transform(test)
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
        spark.stop()
      }
  }
}
// scalastyle:on println
