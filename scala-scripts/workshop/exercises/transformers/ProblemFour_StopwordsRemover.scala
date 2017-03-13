

    //The Data
    val data = Seq(
      (0, Seq("It", "was", "a", "bright", "cold", "day", "in", "April", "and", "the", "clocks", "were", "striking", "thirteen")),
      (1, Seq("The", "sky", "above", "the", "port", "was", "the", "color", "of", "television", "tuned", "to", "a", "dead", "channel")),
      (2, Seq("It", "was", "love", "at", "first", "sight"))
    )

    val dataSet = spark.createDataFrame(data).toDF("id", "raw")

    /**
      * Implement the Stop word remover
      * Use the Input and output column specification
      * Print the resulting output.
      */
// scalastyle:on println
