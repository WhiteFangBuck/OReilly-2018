package com.cloudera.workshop.classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

/**
  * Created by vsingh on 3/3/17.
  */
object ProblemOneLogisticRegression {

    def main(args: Array[String]) {

      Logger.getLogger("org").setLevel(Level.OFF)
      Logger.getLogger("akka").setLevel(Level.OFF)

      /**
        * Read the input data
        */
      var input = "data/Housing.csv"
      if (args.length > 0) {
        input = args(0)
      }

      val spark = SparkSession
        .builder
        .appName("ProblemTwoGradientBoostedTree")
        .master("local")
        .getOrCreate()

      spark.stop()

    }
}
