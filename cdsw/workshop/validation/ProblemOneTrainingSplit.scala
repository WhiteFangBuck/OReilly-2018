import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

/**
 * A simple example demonstrating model selection using TrainValidationSplit.
 *
 */

/**
  * Use the sample Linear Regression Data to demonstrate Model Selection
  */

/**
  * Load the data
  */
val data = spark.read.format("libsvm").load("/data/validation/sample_linear_regression_data.txt")

data.printSchema()
data.show(20,false)

/**
  * Split into training and testing
  */
val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)
training.printSchema()
training.show(20,false)


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
  .addGrid(lr.regParam, Array(0.1, 0.01, 1))
  .addGrid(lr.fitIntercept, Array(true, false))
  .addGrid(lr.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75))
  .build()

/**
  * Do the TrainValidation model initiation
  * 80-20 split
  */
val trainValidationSplit = new TrainValidationSplit()
  .setEstimator(lr)
  .setEvaluator(new RegressionEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setTrainRatio(0.8)

/**
  * Generate the model on the training data
  */
val model = trainValidationSplit.fit(training)

/**
  * Print out the predictions on the test
  */
val result = model.transform(test)
  .select("features", "label", "prediction")
  result.printSchema()
  result.show(20,false)
