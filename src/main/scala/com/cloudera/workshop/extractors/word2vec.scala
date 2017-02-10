package org.cloudera.workshop 

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

object word2vec {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Word2VecExample")
    val sc = new SparkContext(conf)

    val input = sc.textFile("data/mllib/sample_lda_data.txt").map(line => line.split(" ").toSeq)

    val word2vec = new Word2Vec()

    The model maps each word to a unique fixed-size vector.
    val model = word2vec.fit(input)

    val synonyms = model.findSynonyms("1", 5)

    for((synonym, cosineSimilarity) <- synonyms) {
      println(s"$synonym $cosineSimilarity")
    }

    // Save and load model
    model.save(sc, "myModelPath")
    val sameModel = Word2VecModel.load(sc, "myModelPath")

    sc.stop()
  }
}

// scalastyle:on println
