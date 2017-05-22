
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
package org.cloudera.workshop

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
    * Case class is one of mapping incoming data onto the DataFrame columns
    * @param id
    * @param price
    * @param lotsize
    * @param bedrooms
    * @param bathrms
    * @param stories
    * @param driveway
    * @param recroom
    * @param fullbase
    * @param gashw
    * @param airco
    * @param garagepl
    * @param prefarea
    */
  case class X(
                id: String ,price: Double, lotsize: Double, bedrooms: Double,
                bathrms: Double,stories: Double, driveway: String,recroom: String,
                fullbase: String, gashw: String, airco: String, garagepl: Double, prefarea: String)


   Logger.getLogger("org").setLevel(Level.OFF)
   Logger.getLogger("akka").setLevel(Level.OFF)

   /**
     * Read the input data
     */
   var dataset = "data/Housing.csv"

   val spark = SparkSession
      .builder
      .appName("ProblemOneLinearRegression")
      .master("local")
      .getOrCreate()

   /**
     * Create the data frame
     */


   /**
     * Define and Identify the Categorical variables
     */

   /**
     * Initialize the Categorical Varaibles as first state of the pipeline
     */


   /**
     * Initialize the OneHotEncoder as another pipeline stage
     */


   /**
     * Put all the feature columns of the categorical variables together
     */

   /**
     * Initialize the instance for LinearRegression using your choice of solver and number of iterations
     * Experiment with intercepts and different values of regularization parameter
     */


   /**
     * Using cross validation and parameter grid for model tuning
     */


   /**
     * Gather the steps and create the pipeline
     */

   /**
     * Initialize the Cross Validator for model tuning
     */


  /** val tvs = new TrainValidationSplit()
    * .setEstimator( pipeline )
    * .setEvaluator( new RegressionEvaluator()
    * .setLabelCol("price") )
    * .setEstimatorParamMaps(paramGrid)
    * .setTrainRatio(0.75)*/

   /**
     * Split the training and testing data
     */


   /**
     * Fit the model and print out the result
     */

