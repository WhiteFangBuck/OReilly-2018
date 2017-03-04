package com.cloudera.workshop

import org.apache.spark.sql.SparkSession

/**
 * A simple example demonstrating model selection using TrainValidationSplit.
 *
 */
object ProblemOneTrainingSplit {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
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

    /**
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
