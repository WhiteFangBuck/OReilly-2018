    import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

    //Given the following data set, create a dataset with columns labelled as "id" and "category"\

    val data = Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )

    //Step 1:  Create a DataFrame

    //Step 2: Transform to a Dataset

    //Step 3: Initialize a StringIndexer to fit the DataSet

    //Step 4: Convert it back to the original String.

// scalastyle:on println
