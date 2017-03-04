package com.cloudera.workshop

import org.apache.spark.sql.SparkSession

/**
 * A simple example demonstrating model selection using CrossValidator.
 * This example also demonstrates how Pipelines are Estimators.
 *
 */
object ProblemTwoCV {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
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


    /**
      * Use the HashingTF on Tokenizer output
      */

    /**
      * Initialize a logistic regression model
      */

    /**
      * Initialize the pipeline using previous three nodes
      */

    /**
      * Create the parameter builder using various number of features and different values for the regularizer
      */

    /**
      * Initiate a CrossValidator
      * Treate the pipeline as an estimator and a BinaryClassificationEvaluator as testor.
      * NumberofFolds are 2+.
      */

    /**
      * Run cross validation
      */

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

    spark.stop()
  }
}
// scalastyle:on println
