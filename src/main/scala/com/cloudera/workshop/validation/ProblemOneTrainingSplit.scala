package com.cloudera.workshop

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

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
    val data = spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")

    /**
      * Split into training and testing
     */
    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)


    /**
      * Set the number of iterations
      */
    val lr = new LinearRegression()
      .setMaxIter(10)

    /**
      * Set the parametric grid builder
      * For Regularizer
      * For Intercept
      */
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    /**
      * Do the TrainValidation model initiation
      * 80-20 split
      */
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)

    /**
      * Generate the model on the training data
      */
    val model = trainValidationSplit.fit(training)

    /**
      * Print out the predictions on the test
      */
    model.transform(test)
      .select("features", "label", "prediction")
      .show()

    spark.stop()
  }
}
