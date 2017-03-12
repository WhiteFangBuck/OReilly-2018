/**
  * Created by vsingh on 3/11/17.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._

/**
  * Created by vsingh on 3/4/17.
  */


Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)

var inputDir = "data/moviereviews.tsv"
var stopWordFile = "data/stopwords.txt"

val vocabSize: Int = 10000

val df = spark.read.option("header", "true")
    .option("sep", "\t")
    .csv(inputDir)

 /**
   * Clean the HTML Data
   *
   * Hint: Use UDF
   */
 val cleanData = udf((sentiment: String) => sentiment.replaceAll( """<(?!\/?a(?=>|\s.*>))\/?.*?>""", ""))

 val cleanDF = df.withColumn("cleanedReview", cleanData(df.col("review")))

 /**
   * Use RegEx Tokenizer to tokenize the words using several parameters, such as
   *
   * Token Length
   * Tokenization criteria
   * SetGaps or not
   */
  val tokens = new RegexTokenizer()
      .setGaps(false)
      .setPattern("\\w+")
      .setMinTokenLength(4)
      .setInputCol("cleanedReview")
      .setOutputCol("words")
      .transform(cleanDF)

  /**
    * Use stop words to remove or add the words from the list
    * These words will be used for filtering out the words not needed
    */

  val stopwords = spark.sparkContext.textFile(stopWordFile).collect
  val filteredTokens = new StopWordsRemover()
      .setStopWords(stopwords)
      .setCaseSensitive(false)
      .setInputCol("words")
      .setOutputCol("filtered")
      .transform(tokens)

  /**
    * Optionally use NGrams to form the feature vectors
    */

    /* val ngram = new NGram()
      .setInputCol("filtered")
      .setOutputCol("ngrams")
      .transform(filteredTokens)
*/
  /**
    * Use CountVectorizer to generate the numeric vectors
    */

  val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(vocabSize)
      .fit(filteredTokens)

  val countVectors = cvModel
      .transform(filteredTokens)
      .select("id", "sentiment", "features")

  /**
    * Index labels, adding metadata to the label column.
    * // Fit on whole dataset to include all labels in index.
    */
  val labelIndexer = new StringIndexer()
      .setInputCol("sentiment")
      .setOutputCol("indexedLabel")
      .fit(countVectors)

  val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(countVectors)

  /**
    * Split the data into training and test sets (30% held out for testing)
    */
  val Array(trainingData, testData) = countVectors.randomSplit(Array(0.7, 0.3))

  val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(200)

  val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

  /**
    * Use the pipeline to fit in the stages
    */
  val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

  /**
    * Train model. This also runs the indexers.
    */
  val model = pipeline.fit(trainingData)

  /**
    * Make predictions
    */
  val predictions = model.transform(testData)

  /**
    * Display Result
    */
  predictions.select("predictedLabel", "sentiment", "features").show(30)

  /**
    * Select (prediction, true label) and compute test error
    */

  val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)


  /**
    * Print the error
    */
  println("Test Error = " + (1.0 - accuracy))
  val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
  println("Learned classification forest model:\n" + rfModel.toDebugString)
