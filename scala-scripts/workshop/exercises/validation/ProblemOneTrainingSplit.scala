import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

Logger.getRootLogger.setLevel(Level.OFF)
Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)

/**
  * Use the sample Linear Regression Data to demonstrate Model Selection
  * IMPORTANT: Uncomment the dataset below
  */

var dataset = "UNCOMMENT_YOUR_DATASET"

// If you are using spark-shell, uncomment this line
// dataset = "data/validation/sample_linear_regression_data.txt"

// If you are using CDSW, uncomment this line
// dataset = "/data/validation/sample_linear_regression_data.txt"

/**
  * Load the data
  */
val data = spark.read.format("libsvm").load(dataset)


/**
  * Split into training and testing
  */
val Array(training, test) = data.randomSplit(Array(0.9, 0.1), seed = 12345)
training.printSchema()
training.show()

/**
  * Initialize LinearRegression and
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
