import org.apache.spark.ml.feature._

{{
    // Input data: Each row is a bag of words from a sentence or document.
    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "It was a bright cold day in April, and the clocks were striking thirteen."),
      (0.0, "The sky above the port was the color of television, tuned to a dead channel."),
      (1.0, "It was love at first sight.")
    )).toDF("label", "sentence")
    
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2Vec.fit(wordsData)

    val result = model.transform(wordsData)
    result.show(false)  
}}
