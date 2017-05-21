{{
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.{col, udf}

/**
  * Combine above using VectorIndexer
  */
val libSVMData = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val vIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexed")
      .setMaxCategories(10)

val indexerModel = vIndexer.fit(libSVMData)

val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
    println(s"Chose ${categoricalFeatures.size} categorical features: " +
      categoricalFeatures.mkString(", "))

// Create new column "indexed" with categorical values transformed to indices
val indexedData = indexerModel.transform(libSVMData)
    indexedData.show()
    
    }}
