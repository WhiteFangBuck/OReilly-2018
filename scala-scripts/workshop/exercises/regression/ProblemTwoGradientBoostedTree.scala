package com.cloudera.workshop.regression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

/**
  * Created by vsingh on 3/3/17.
  */

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    /**
      * Read the input data
      */
    var input = "/data/Housing.csv"

