/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.cloudera.workshop

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{CountVectorizer, NGram, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.clustering._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql._
/**
  * Infer the cluster topics on a set of 20 newsgroup data.
  *
  * The input text is text files, corresponding to emails in the newsgroup.
  * Each text file corresponds to one document.
  *
  *
  */
object TopicModelingExample {

	def main (args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    var inputDir = "data/topicmodeling/newsgroup_20/"
    var stopWordFile = "data/topicmodeling/stopwords.txt"

    if(args.length > 1) {
      inputDir = args(0)
      stopWordFile = args(1)
    }
    
    val spark = SparkSession
      .builder
      .appName("TopicModelingExample")
      .master("local")
      .getOrCreate()
      
    import spark.implicits._

    val numTopics: Int = 10
    val maxIterations: Int = 100
    val vocabSize: Int = 10000

    val rawTextRDD = spark.sparkContext.wholeTextFiles(inputDir).map(_._2)
    val docDF = rawTextRDD
                  .zipWithIndex.toDF("text", "docId")
    val tokens = new RegexTokenizer()
                  .setGaps(false)
                  .setPattern("\\w+")
                  .setMinTokenLength(4)
                  .setInputCol("text")
                  .setOutputCol("words")
                  .transform(docDF)

    val stopwords = spark.sparkContext.textFile(stopWordFile).collect
    val filteredTokens = new StopWordsRemover()
                          .setStopWords(stopwords)
                          .setCaseSensitive(false)
                          .setInputCol("words")
                          .setOutputCol("filtered")
                          .transform(tokens)
		
   val ngram = new NGram()
                  .setInputCol("filtered")
                  .setOutputCol("ngrams")
                  .transform(filteredTokens)


    val cvModel = new CountVectorizer()
                    .setInputCol("ngrams")
                    .setOutputCol("features")
                    .setVocabSize(vocabSize)
                    .fit(ngram)

    val countVectors = cvModel
                        .transform(ngram)
                        .select("docId", "features")

    val lda = new LDA()
                  .setOptimizer("online")
                  .setK(numTopics)
                  .setMaxIter(maxIterations)

    val startTime = System.nanoTime()
    val ldaModel = lda.fit(countVectors.repartition(20))
    val elapsed = (System.nanoTime() - startTime) / 1e9

    println(s"Finished training LDA model.  Summary:")
    println(s"Training time (sec)\t$elapsed")
    println(s"==========")

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10).coalesce(1)
    val vocabArray = cvModel.vocabulary
    
    for(i <- topicIndices) { println(s"Topic ${i(0)}")
     val a: Array[Int] = i(1).asInstanceOf[scala.collection.mutable.WrappedArray[Int]].toSeq.toArray
     val b: Array[Double] = i(2).asInstanceOf[scala.collection.mutable.WrappedArray[Double]].toSeq.toArray
     a.map(vocabArray(_)).zip(b).foreach { case (term, weight) => println(s"$term\t$weight") }
     println(s"==================")
    }
   // val topics = topicIndices.map {
   //               case (terms, termWeights) =>
   //               terms.map(vocabArray(_)).zip(termWeights)
   // }

    //topics.zipWithIndex.foreach {
      //    case (topic, i) => println(s"TOPIC $i")
      //    topic.foreach { case (term, weight) => println(s"$term\t$weight") }
      //    println(s"==========")
    //}
  }
}
