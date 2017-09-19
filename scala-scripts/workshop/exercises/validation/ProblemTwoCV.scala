import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}


Logger.getRootLogger.setLevel(Level.OFF)
Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)

/**
  * A simple example demonstrating model selection using CrossValidator.
  * This example also demonstrates how Pipelines are Estimators.
  *
  */

/**
  * This is your training data
  */

val dataset = "/data/validation/farm-ads.txt"

val schema = StructType(Array(
  StructField("label", DoubleType, true),
  StructField("text", StringType, true)))

val originalDF = spark.read.
  format("csv").
  option("header","false").
  schema(schema).
  load(dataset)


val trainingDF = originalDF.withColumn("id",
  monotonically_increasing_id())

trainingDF.printSchema()
trainingDF.show()


/**
  * Tokenize the \"text"\ column
  * Start of the pipeline
  */

/**
  * Use the HashingTF on Tokenizer output
  */

/**
  * Apply IDF
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
  * Sample Test DataSet
  */
val test = spark.createDataFrame(Seq(
  (111100005L,"ad-animal"),
  (111100006L,"The airplane flew low"),
  (111100008L,"ad-supplement"),
  (111100009L,"circulatory support immune support joint support vitamin mineral weight loss product horse total health "),
  (1111000010L,"ad-jerry ad-bruckheimer ad-chase ad-premier ad-sept ad-th ad-clip ad-bruckheimer ad-chase page found")
))toDF("id", "text")

/**
  * Make predictions on the test documents
  */
