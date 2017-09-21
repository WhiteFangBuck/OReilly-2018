import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.{col, udf}

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
var stopWordFile = "/data/stopwords.txt"
val stopwords = spark.sparkContext.textFile(stopWordFile).collect
val remover = new StopWordsRemover().
      setStopWords(stopwords).
      setCaseSensitive(false).
      setInputCol("words").
      setOutputCol("filtered")

remover.transform(regexTokenized).show(false)
      
