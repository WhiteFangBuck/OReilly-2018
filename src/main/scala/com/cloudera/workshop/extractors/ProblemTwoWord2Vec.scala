package org.cloudera.workshop 

import org.apache.spark.sql.SparkSession

object ProblemTwoWord2Vec {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("ProblemTwoWord2Vec")
      .getOrCreate()

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

  }
}

// scalastyle:on println
