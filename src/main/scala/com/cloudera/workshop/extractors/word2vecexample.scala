package org.cloudera.workshop 

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

object word2vecexample {

  def main(args: Array[String]): Unit = {

    /**
      * Read in the data to generate and display the word2vec model
       */
    /**
      * Create the Dataframe to be used for
      */

    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "It was a bright cold day in April, and the clocks were striking thirteen."),
      (0.0, "The sky above the port was the color of television, tuned to a dead channel."),
      (1.0, "It was love at first sight.")
    )).toDF("label", "sentence")

    /**
      * Generate the word2vec model
      */

    sc.stop()
  }
}

// scalastyle:on println
