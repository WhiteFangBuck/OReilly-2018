package org.cloudera.workshop

import org.apache.spark.sql.SparkSession

object countvectorizerexample {

  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("CountVectorizerExample")
      .getOrCreate()

    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "It was a bright cold day in April, and the clocks were striking thirteen."),
      (0.0, "The sky above the port was the color of television, tuned to a dead channel."),
      (1.0, "It was love at first sight.")
    )).toDF("label", "sentence")

    /**
      *  Fit a CountVectorizer Model from the given corpus
       */

    /**
      * Try this by defining a-priori vocabulary
      */


    spark.stop()
  }
}
// scalastyle:on println
