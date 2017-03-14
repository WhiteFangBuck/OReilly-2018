import org.apache.log4j._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.sql.functions._

Logger.getRootLogger.setLevel(Level.OFF)
Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)

/**
  * As simple K-Means clustering example
  */


/**
  * Create dataframe using csv method.
  * IMPORTANT: Uncomment the dataset below
  */

var dataset = "UNCOMMENT_YOUR_DATASET"
// If you are using spark-shell, uncomment this line
// dataset = "data/kmeans/flightinfo/flights_nofeatures.csv"

// If you are using CDSW, uncomment this line
// dataset = "/data/kmeans/flightinfo/flights_nofeatures.csv"

val inputData = spark.read
  .option("header","true")
  .option("inferSchema","true").csv(dataset)



/**
  * Transform Day into Something Usable
  * We are expanding each day into a new feature
  * 1 is the value if its that day, 0 if any other day
  *
  */


/**
  * Transform Time into something usable
  * We are taking time as a fraction of the day
  * That gives us a very good feature to cluster on
  *
  */

/**
  * UDF to convert to Int
  *
  */


/**
  * Use VectorAssembler to assemble feature vector
  * From relevant columns
  *
  */


/**
  * Scale my features using MinMaxScaler
  */


/**
  * Trains a k-means model
  */

/**
  * Run the model to generate Cluster IDs
  */
