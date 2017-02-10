package org.cloudera.workshop

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

object tfidfexample {
def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("TfIdfExample")
      .getOrCreate()

    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
  
  //Both HashingTF and CountVectorizer can be used to generate the term frequency vectors
  //https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.HashingTF
  //https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.feature.CountVectorizer
  
  //HashingTF is a Transformer which takes sets of terms and converts those sets into fixed-length feature vectors.
  //CountVectorizer converts text documents to vectors of term counts.

    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    //IDF is an Estimator which is fit on a dataset and produces an IDFModel. 
  //The IDFModel takes feature vectors (generally created from HashingTF or CountVectorizer) and scales each column. 
  //Intuitively, it down-weights columns which appear frequently in a corpus.
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()

    spark.stop()
  }
}
// scalastyle:on println
