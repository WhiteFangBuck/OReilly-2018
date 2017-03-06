package com.cloudera.workshop.classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.sql.SparkSession

/**
  * Created by vsingh on 3/4/17.
  */
object ProblemFourRandomForestClassifier {

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    var inputDir = "data/labeledTrainData.tsv"
    var stopWordFile = "data/stopwords.txt"

    val vocabSize: Int = 10000

    if (args.length > 1) {
      inputDir = args(0)
      stopWordFile = args(1)
    }

    val spark = SparkSession
      .builder
      .appName("LDASolution")
      .master("local")
      .getOrCreate()

    /**
      * Load the Data using spark csv
      */


    /**
      * Clean the HTML Data
      *
      * Hint: Use UDF
      */


    /**
      * Use RegEx Tokenizer to tokenize the words using several parameters, such as
      *
      * Token Length
      * Tokenization criteria
      * SetGaps or not
      */

    /**
      * Use stop words to remove or add the words from the list
      * These words will be used for filtering out the words not needed
      */

    /**
      * Optionally use NGrams to form the feature vectors
      */

    /**
      * Use CountVectorizer to generate the numeric vectors
      */


    /**
      * Index labels, adding metadata to the label column.
      */

    /**
      * Split the data into training and test sets (30% held out for testing)
      */
    //val Array(trainingData, testData) = countVectors.randomSplit(Array(0.7, 0.3))

    /**
      * Initialize the Random Forest Tree
      */
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(200)

    /**
      * Use the pipeline to fit in the stages
      */

    /**
      * Train model. This also runs the indexers.
      */

    /**
      * Make predictions
      */

    /**
      * Display Result
      */

    /**
      * Select (prediction, true label) and compute test error
      */

    /**
      * Print the error
      */
  }
}
