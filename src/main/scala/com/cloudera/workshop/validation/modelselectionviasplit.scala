package com.cloudera.workshop

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

/**
 * A simple example demonstrating model selection using TrainValidationSplit.
 *
 */
object modelselectionviasplit {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("ModelSelectionViaTrainValidationSplitExample")
      .getOrCreate()

    // Prepare training and test data.
    val data = spark.read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")
    val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)

    val lr = new LinearRegression()
        .setMaxIter(10)

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // TrainValidationSplit will try all combinations of values and determine best model using
    // the evaluator.
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    // In this case the estimator is simply the linear regression.
    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)

    // Run train validation split, and choose the best set of parameters.
    val model = trainValidationSplit.fit(training)

    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    model.transform(test)
      .select("features", "label", "prediction")
      .show()

    spark.stop()
  }
}
