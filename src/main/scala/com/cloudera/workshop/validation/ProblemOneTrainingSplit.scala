package com.cloudera.workshop.validation

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

/**
 * A simple example demonstrating model selection using TrainValidationSplit.
 *
 */
object ProblemOneTrainingSplit {
  Logger.getRootLogger.setLevel(Level.OFF)
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[2]")
      .appName("ProblemOneTrainingSplit")
      .getOrCreate()

    /**
      * Use the sample Linear Regression Data to demonstrate Model Selection
      */

    /**
      * Load the data
     */
    val data = spark.read.format("libsvm").load("data/validation/sample_linear_regression_data.txt")

    data.printSchema()
    data.show()

    /**
      * Split into training and testing
     */
    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)
    training.printSchema()
    training.show()


    /**
      * Create LinearRegression Object and
      * Set the number of iterations
      */

    /**
      * Set the parametric grid builder
      * For Regularizer
      * For Intercept
      */


    /**
      * Do the TrainValidation model initiation
      * 80-20 split
      */


    /**
      * Generate the model on the training data
      */

    /**
      * Print out the predictions on the test
      */


    spark.stop()
  }
}
