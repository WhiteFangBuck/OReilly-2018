
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{abs, col}


/**
* Case class is one of mapping incoming data onto the DataFrame columns
*/
case class X(
              id: String ,price: Double, lotsize: Double, bedrooms: Double,
              bathrms: Double,stories: Double, driveway: String,recroom: String,
              fullbase: String, gashw: String, airco: String, garagepl: Double, prefarea: String)


Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)

/**
    * Create dataframe using csv method.
*/

var dataset =  "/data/Housing.csv"

import spark.implicits._

  /**
    * Create the data frame
    */

val data = spark.sparkContext.textFile(dataset).
    map(_.split(",")).
    map( x => ( X(
      x(0), x(1).toDouble, x(2).toDouble, x(3).toDouble, x(4).toDouble, x(5).toDouble,
      x(6), x(7), x(8), x(9), x(10), x(11).toDouble, x(12) ))).
    toDF()

data.show(20)

/**
    * Define and Identify the Categorical variables
    */
val categoricalVariables = Array("driveway","recroom", "fullbase", "gashw", "airco", "prefarea")

/**
    * Initialize the Categorical Varaibles as first state of the pipeline
  */

val categoricalIndexers: Array[org.apache.spark.ml.PipelineStage] =
  categoricalVariables.map(i => new StringIndexer().
    setInputCol(i).setOutputCol(i+"Index"))

  /**
    * Initialize the OneHotEncoder as another pipeline stage
    */

val categoricalEncoders: Array[org.apache.spark.ml.PipelineStage] =
  categoricalVariables.map(e => new OneHotEncoder().
    setInputCol(e + "Index").setOutputCol(e + "Vec"))

/**
    * Put all the feature columns of the categorical variables together
    */

val assembler = new VectorAssembler().
    setInputCols( Array(
      "lotsize", "bedrooms", "bathrms", "stories",
      "garagepl","drivewayVec", "recroomVec", "fullbaseVec",
      "gashwVec","aircoVec", "prefareaVec")).
    setOutputCol("features")

  /**
    * Initialize the instance for LinearRegression using your choice of solver and number of iterations
    * Experiment with intercepts and different values of regularization parameter
    */


val lr = new LinearRegression().
    setLabelCol("price").
    setFeaturesCol("features").
    setRegParam(0.1).
    setMaxIter(100).
    setSolver("l-bfgs")

/**
  * Gather the steps and create the pipeline
  */
val steps = categoricalIndexers ++
  categoricalEncoders ++
  Array(assembler, lr)

val pipeline = new Pipeline().setStages(steps)

/**
  * Split the data into training and test
  */
val Array(training, test) = data.randomSplit(Array(0.75, 0.25), seed = 12345)

/**
  * Fit the model and print out the result
  */

val model = pipeline.fit {
  training
}

val holdout = model.transform(test)
holdout.show(20)

val prediction = holdout.select("prediction", "price").orderBy(abs(col("prediction")-col("price")))
prediction.show(20)


val rm = new RegressionMetrics(prediction.rdd.map{
  x =>  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])
})
println(s"RMSE = ${rm.rootMeanSquaredError}")
println(s"R-squared = ${rm.r2}")
