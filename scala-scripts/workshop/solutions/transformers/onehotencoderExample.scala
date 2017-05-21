{{

import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.{col, udf}

val df = spark.createDataFrame(Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )).toDF("id", "category")

/**
  * StringIndexer Example
  */
val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)

 val indexed = indexer.transform(df)

 println(s"Transformed string column '${indexer.getInputCol}' " +
      s"to indexed column '${indexer.getOutputCol}'")
    indexed.show()

 val inputColSchema = indexed.schema(indexer.getOutputCol)
    println(s"StringIndexer will store labels in output column metadata: " +
      s"${Attribute.fromStructField(inputColSchema).toString}\n")

 /**
   * IndexToString Example
   */

 val converter = new IndexToString()
      .setInputCol("categoryIndex")
      .setOutputCol("originalCategory")

 val converted = converter.transform(indexed)

 println(s"Transformed indexed column '${converter.getInputCol}' back to original string " +
      s"column '${converter.getOutputCol}' using labels in metadata")
    converted.select("id", "categoryIndex", "originalCategory").show()

 val encoder = new OneHotEncoder()
      .setInputCol("categoryIndex")
      .setOutputCol("categoryVec")


val encoded = encoder.transform(indexed)
    encoded.show()
    
    }}
