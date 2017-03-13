package com.cloudera.workshop.clustering

import org.apache.log4j._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.sql.functions._

object ProblemOneKMeans{
  Logger.getRootLogger.setLevel(Level.OFF)
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    val session = org.apache.spark.sql.SparkSession.builder().
      master("local[4]")
      .appName("ProblemOneKMeans")
      .getOrCreate()

    /**
      * Create dataframe using csv method.
      *
      */    val dataset = "data/kmeans/flightinfo/flights_nofeatures.csv"
    val inputData = session.read
        .option("header","true")
        .option("inferSchema","true").csv(dataset)


    /**
      * Transform Day into Something Usable
      * We are expanding each day into a new feature
      * 1 is the value if its that day, 0 if any other day
      *
      */


    /**
      * Transform Time into something usable
      * We are taking time as a fraction of the day
      * That gives us a very good feature to cluster on
      *
      */

    /**
      * UDF to convert to Int
      *
      */



    /**
      * Use VectorAssembler to assemble feature vector
      * From relevant columns
      *
      */


    /**
      * Scale my features using MinMaxScaler
      */

    /**
      * Trains a k-means model
      */


    /**
      * Run the model to generate Cluster IDs
      */



    session.stop()
  }
}
