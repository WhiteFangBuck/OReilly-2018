package com.cloudera.workshop.solutions

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature.{CountVectorizer, NGram, RegexTokenizer, StopWordsRemover}
import org.apache.spark.sql.SparkSession

/**
  * Created by vsingh on 3/3/17.
  */
object LDASolution {


  def main (args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    var inputDir = "data/topicmodeling/newsgroup_20/"
    var stopWordFile = "data/topicmodeling/stopwords.txt"

    if(args.length > 1) {
      inputDir = args(0)
      stopWordFile = args(1)
    }

    val spark = SparkSession
      .builder
      .appName("LDASolution")
      .master("local")
      .getOrCreate()

    import spark.implicits._

    /**
      * There are three hyperparameters here
      */

    val numTopics: Int = 10
    val maxIterations: Int = 100
    val vocabSize: Int = 10000

    /**
      * Read the files are the directory level
      *
      * Or
      *
      * Find a way to read all the files and attach an id to the files read.
      */

    val rawTextRDD = spark.sparkContext.wholeTextFiles(inputDir).map(_._2)
    val docDF = rawTextRDD
      .zipWithIndex.toDF("text", "docId")

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
      .setInputCol("text")
      .setOutputCol("words")
      .transform(docDF)

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

    val ngram = new NGram()
      .setInputCol("filtered")
      .setOutputCol("ngrams")
      .transform(filteredTokens)

    /**
      * Use CountVectorizer to generate the numeric vectors
      */

    val cvModel = new CountVectorizer()
      .setInputCol("ngrams")
      .setOutputCol("features")
      .setVocabSize(vocabSize)
      .fit(ngram)

    val countVectors = cvModel
      .transform(ngram)
      .select("docId", "features")

    /**
      * Initialize the LDA
      * Either use EM optimizer or online optimizer.
      */

    val lda = new LDA()
      .setOptimizer("online")
      .setK(numTopics)
      .setMaxIter(maxIterations)

    /**
      * Print out the Word to Topic probabilities
      */

    val startTime = System.nanoTime()
    val ldaModel = lda.fit(countVectors.repartition(20))
    val elapsed = (System.nanoTime() - startTime) / 1e9

    println(s"Finished training LDA model.  Summary:")
    println(s"Training time (sec)\t$elapsed")
    println(s"==========")

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10).coalesce(1)
    val vocabArray = cvModel.vocabulary

    for(i <- topicIndices) { println(s"Topic ${i(0)}")
      val a: Array[Int] = i(1).asInstanceOf[scala.collection.mutable.WrappedArray[Int]].toSeq.toArray
      val b: Array[Double] = i(2).asInstanceOf[scala.collection.mutable.WrappedArray[Double]].toSeq.toArray
      a.map(vocabArray(_)).zip(b).foreach { case (term, weight) => println(s"$term\t$weight") }
      println(s"==================")
    }

    spark.stop()
    // val topics = topicIndices.map {
    //               case (terms, termWeights) =>
    //               terms.map(vocabArray(_)).zip(termWeights)
    // }

    //topics.zipWithIndex.foreach {
    //    case (topic, i) => println(s"TOPIC $i")
    //    topic.foreach { case (term, weight) => println(s"$term\t$weight") }
    //    println(s"==========")
    //}
  }

}
