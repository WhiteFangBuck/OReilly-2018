package org.cloudera.workshop

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.sql.functions._

object clustering {
  Logger.getRootLogger.setLevel(Level.OFF)
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    val session = org.apache.spark.sql.SparkSession.builder().
      master("local[4]")
      .appName("IRS Public Charities")
      .getOrCreate()



    val df = session.read
      .format("com.databricks.spark.xml")
      .option("rowTag", "IRS990")
      .load("resources/990")

    // Select your columns here
    // val chosenData = df.select().cache()


    //  Convert String to Numeric
    //  Helper function to translate booleans
    val toNum = udf { (x: String) => if (x == null) -1
    else if (x.equals("false")) 0
    else if (x.equals("true")) 1
    else x.hashCode
    }

      // Add boolean columns with translation if required
      //val numericDF = chosenData.withColumn().na.fill(-1)

      // Assemble Feature columns
      //val assembler = new VectorAssembler()
      //  .setInputCols(Array()).setOutputCol("features")

      // Add a features columns, using assembler above
      // val featurizedDF =


      // Normalize the features
      val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)

  //  val normalizedDF = normalizer.

    val kmeans = new KMeans()
      .setK(20)
      .setFeaturesCol("normFeatures")
      .setPredictionCol("clusterId")
  //  val model = kmeans.fit(normalizedDF)


    //val predictedCluster = model.transform(normalizedDF)
    //predictedCluster.printSchema()
    //predictedCluster.show()


    session.stop()
  }
}
