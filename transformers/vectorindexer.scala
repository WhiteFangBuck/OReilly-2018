
import org.apache.spark.ml.feature.VectorIndexer

import org.apache.spark.sql.SparkSession

object vectorindexer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("VectorIndexerExample")
      .getOrCreate()

    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val indexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexed")
      .setMaxCategories(10)

    val indexerModel = indexer.fit(data)

    val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
    println(s"Chose ${categoricalFeatures.size} categorical features: " +
      categoricalFeatures.mkString(", "))

    // Create new column "indexed" with categorical values transformed to indices
    val indexedData = indexerModel.transform(data)
    indexedData.show()

    spark.stop()
  }
}
// scalastyle:on println
