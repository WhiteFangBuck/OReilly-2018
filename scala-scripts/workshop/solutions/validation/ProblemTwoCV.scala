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

/**
  * A simple example demonstrating model selection using CrossValidator.
  * This example also demonstrates how Pipelines are Estimators.
  *
  */

/**
  * This is your training data
  * IMPORTANT: Uncomment the dataset below
  */

// If you are using spark-shell, uncomment this line
//val dataset = "data/validation/farm-ads.txt"

// If you are using CDSW, uncomment this line
//val dataset = "/data/validation/farm-ads.txt"

val schema = StructType(Array(
  StructField("label", DoubleType, true),
  StructField("text", StringType, true)))

val originalDF = spark.read
  .format("csv")
  .option("header","false")
  .schema(schema)
  .load(dataset)


val trainingDF = originalDF.withColumn("id",
  monotonically_increasing_id())

trainingDF.printSchema()
trainingDF.show()


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
  .setOutputCol("rawFeatures")

/**
  * Apply IDF
  */
val idf = new IDF()
  .setInputCol("rawFeatures")
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
  .setStages(Array(tokenizer, hashingTF, idf, lr))

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
val cvModel = cv.fit(trainingDF)

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
cvModel.transform(test)
  .select("id", "text", "probability", "prediction")
  .collect()
  .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
println(s"($id, $text) --> prob=$prob, prediction=$prediction")}