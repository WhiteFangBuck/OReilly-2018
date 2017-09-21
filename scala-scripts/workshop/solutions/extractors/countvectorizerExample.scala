import org.apache.spark.ml.feature._

    /**
      * Create the Dataframe to be used for
      */

    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "It was a bright cold day in April, and the clocks were striking thirteen."),
      (0.0, "The sky above the port was the color of television, tuned to a dead channel."),
      (1.0, "It was love at first sight.")
    )).toDF("label", "sentence")

  /**
  * Use RegEx Tokenizer to tokenize the words using several parameters, such as
  *
  * Token Length
  * Tokenization criteria
  * SetGaps or not
  */
   val wordsData = new RegexTokenizer().
      setGaps(false). 
      setPattern("\\w+"). 
      setMinTokenLength(4). 
      setInputCol("sentence"). 
      setOutputCol("words"). 
      transform(sentenceData) 
  
    /**
      * Work on CountVectorizer
      *
      * Experiment with changing the value of the vocabSize
      * Experiment with changing the value of minDF
      */

    val cvModel: CountVectorizerModel = new CountVectorizer().
      setInputCol("words").
      setOutputCol("features").
      setVocabSize(40).
      setMinDF(1).
      fit(wordsData)
      
    cvModel.transform(wordsData).show(10)

   /**
   * alternatively, define CountVectorizerModel with a-priori vocabulary
   */
    val cvm = new CountVectorizerModel(Array("a", "b", "c")).
      setInputCol("words").
      setOutputCol("features")

    cvModel.transform(wordsData).show(false)
