package com.cloudera.workshop

import org.apache.spark.sql.SparkSession

object onehotencoderexample {


  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("OneHotEncoderExample")
      .getOrCreate()

    val data = Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )

    //Do the one hot encoding.
    //Hint: Build upon StringIndexer from the previous example

    spark.stop()
  }
}
// scalastyle:on println
