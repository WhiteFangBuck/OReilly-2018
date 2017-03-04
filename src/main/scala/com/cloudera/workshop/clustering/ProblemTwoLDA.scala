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
import org.apache.spark.sql._
/**
  * Infer the cluster topics on a set of 20 newsgroup data.
  *
  * The input text is text files, corresponding to emails in the newsgroup.
  * Each text file corresponds to one document.
  *
  *
  */
object ProblemTwoLDA {

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
      .appName("ProblemTwoLDA")
      .master("local")
      .getOrCreate()

    /**
      * There are three hyperparameters here
      */

    val numTopics: Int = 10
    val maxIterations: Int = 100
    val vocabSize: Int = 10000

    /**
      * Read the files are the directory level
      *
      * Or
      *
      * Find a way to read all the files and attach an id to the files read.
      */


    /**
      * Use RegEx Tokenizer to tokenize the words using several parameters, such as
      *
      * Token Length
      * Tokenization criteria
      * SetGaps or not
      */

    /**
      * Use stop words to remove or add the words from the list
      * These words will be used for filtering out the words not needed
      */

    /**
      * Optionally use NGrams to form the feature vectors
      */

    /**
      * Use CountVectorizer to generate the numeric vectors
      */

    /**
      * Initialize the LDA
      * Either use EM optimizer or online optimizer.
      */

    /**
      * Print out the Word to Topic probabilities
      */

  }
}
