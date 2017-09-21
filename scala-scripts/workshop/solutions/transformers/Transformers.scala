
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
val indexer = new StringIndexer().
      setInputCol("category").
      setOutputCol("categoryIndex").
      fit(df)

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

 val converter = new IndexToString().
      setInputCol("categoryIndex").
      setOutputCol("originalCategory")

 val converted = converter.transform(indexed)

 println(s"Transformed indexed column '${converter.getInputCol}' back to original string " +
      s"column '${converter.getOutputCol}' using labels in metadata")
    converted.select("id", "categoryIndex", "originalCategory").show()

 val encoder = new OneHotEncoder().
      setInputCol("categoryIndex").
      setOutputCol("categoryVec")


val encoded = encoder.transform(indexed)
    encoded.show()


/**
  * Combine above using VectorIndexer
  */
val libSVMData = spark.read.format("libsvm").load("/data/mllib/sample_libsvm_data.txt")

val vIndexer = new VectorIndexer().
      setInputCol("features").
      setOutputCol("indexed").
      setMaxCategories(10)

val indexerModel = vIndexer.fit(libSVMData)


val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
    println(s"Chose ${categoricalFeatures.size} categorical features: " +
      categoricalFeatures.mkString(", "))

// Create new column "indexed" with categorical values transformed to indices
val indexedData = indexerModel.transform(libSVMData)
    indexedData.show()

/**
  * Tokenizer
  */
val data = Seq(
      (0, " It was a bright cold day in April, and the clocks were striking thirteen."),
      (1, "The sky above the port was the color of television, tuned to a dead channel."),
      (2, "It was love at first sight.")
    )

val sentenceDataFrame = spark.createDataFrame(data).toDF("id", "sentence")

val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
val regexTokenizer = new RegexTokenizer().
      setInputCol("sentence").
      setOutputCol("words").
      setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

val countTokens = udf { (words: Seq[String]) => words.length }

val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence", "words").
      withColumn("tokens", countTokens(col("words"))).show(false)

val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words").
      withColumn("tokens", countTokens(col("words"))).show(false)

/**
  * StopWords Removal Example
  */

val remover = new StopWordsRemover().
      setInputCol("words").
      setOutputCol("filtered")

remover.transform(regexTokenized).show(false)
