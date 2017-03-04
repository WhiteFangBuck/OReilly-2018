package org.cloudera.workshop

import org.apache.spark.sql.SparkSession

object tfidfexample {

def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("TfIdfExample")
      .getOrCreate()

    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "It was a bright cold day in April, and the clocks were striking thirteen."),
      (0.0, "The sky above the port was the color of television, tuned to a dead channel."),
      (1.0, "It was love at first sight.")
    )).toDF("label", "sentence")

  /**
    * Tokenize the words
    */

  /**
    * Use HashingTF and/or CountVectorizer to generate the IDF
    *
    * HashingTF: This is a transformer that takes a set of terms and converts those sets into fixed-length feature vectors.
    *
    * CountVectorizer: Converts text documents to vectors of term documents.
     */

  /**
    * Generate the IDF Model
    */

  /**
    * Show the transformed data
    */

    spark.stop()
  }
}
// scalastyle:on println
