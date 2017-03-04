package org.cloudera.workshop

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{Normalizer, VectorAssembler}
import org.apache.spark.sql.functions._

object KmeansSolution {
  Logger.getRootLogger.setLevel(Level.OFF)
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def main(args: Array[String]) {

    val session = org.apache.spark.sql.SparkSession.builder().
      master("local[4]")
      .appName("IRS Public Charities")
      .getOrCreate()



    val df = session.read
      .format("com.databricks.spark.xml")
      .option("rowTag", "IRS990")
      .load("resources/990")

    df.schema
    //df.show(12)

    val chosenData = df.select(
      df("GrossReceipts").alias("f1"),
      df("Organization501c").getField("_VALUE").alias("f2"),
      df("TotalNbrEmployees").alias("f3"),
      df("NbrVotingMembersGoverningBody").alias("f4"),
      df("NbrIndependentVotingMembers").alias("f5"),
      df("TotalNbrVolunteers").alias("f6"),
      df("TotalGrossUBI").alias("f7"),
      df("NetUnrelatedBusinessTxblIncome").alias("f8"),
      df("ContributionsGrantsPriorYear").alias("f9"),
      df("ContributionsGrantsCurrentYear").alias("f10"),
      df("ProgramServiceRevenuePriorYear").alias("f11"),
      df("ProgramServiceRevenueCY").alias("f12"),
      df("InvestmentIncomeCurrentYear").alias("f13"),
      df("OtherRevenuePriorYear").alias("f14"),
      df("OtherRevenueCurrentYear").alias("f15"),
      df("TotalRevenuePriorYear").alias("f16"),
      df("TotalRevenueCurrentYear").alias("f17"),
      df("GrantsAndSimilarAmntsCY").alias("f18"),
      df("BenefitsPaidToMembersCY").alias("f19"),
      df("SalariesEtcCurrentYear").alias("f20"),
      df("TotalProfFundrsngExpCY").alias("f21"),
      df("TotalFundrsngExpCurrentYear").alias("f22"),
      df("OtherExpensePriorYear").alias("f23"),
      df("OtherExpensesCurrentYear").alias("f24"),
      df("TotalExpensesPriorYear").alias("f25"),
      df("TotalExpensesCurrentYear").alias("f26"),
      df("TotalAssetsBOY").alias("f27"),
      df("TotalAssetsEOY").alias("f28"),
      df("TotalLiabilitiesBOY").alias("f29"),
      df("TotalLiabilitiesEOY").alias("f30"),
      df("NetAssetsOrFundBalancesBOY").alias("f31"),
      df("NetAssetsOrFundBalancesEOY").alias("f32"),
      df("School").getField("_VALUE").alias("f33"),
      df("ForeignOffice").alias("f34"),
      df("ForeignActivities").getField("_VALUE").alias("f35"),
      df("MoreThan5000KToOrganizations").getField("_VALUE").alias("f36"),
      df("MoreThan5000KToIndividuals").getField("_VALUE").alias("f37"),
      df("ProfessionalFundraising").getField("_VALUE").alias("f38"),
      df("FundraisingActivities").getField("_VALUE").alias("f39"),
      df("Gaming").getField("_VALUE").alias("f40"),
      df("Hospital").getField("_VALUE").alias("f41"),
      df("GrantsToOrganizations").getField("_VALUE").alias("f42"),
      df("GrantsToIndividuals").getField("_VALUE").alias("f43"),
      df("ScheduleJRequired").getField("_VALUE").alias("f44"),
      df("TaxExemptBonds").getField("_VALUE").alias("f45"),
      df("OfficerEntityWithBsnssRltnshp").getField("_VALUE").alias("f46"),
      df("Terminated").getField("_VALUE").alias("f47"),
      df("PartialLiquidation").getField("_VALUE").alias("f48"),
      df("MembersOrStockholders").alias("f49"),
      df("ElectionOfBoardMembers").alias("f50")).cache()

    chosenData.schema
    chosenData.show(20)

    //  Convert String to Numeric
    val toNum = udf { (x: String) => if (x == null) -1
    else if (x.equals("false")) 0
    else if (x.equals("true")) 1
    else x.hashCode
    }

    val numericDF = chosenData.withColumn("f52", toNum(chosenData("f2"))).drop("f2")
      .withColumn("f53", toNum(chosenData("f33"))).drop("f33")
      .withColumn("f54", toNum(chosenData("f34"))).drop("f34")
      .withColumn("f55", toNum(chosenData("f35"))).drop("f35")
      .withColumn("f56", toNum(chosenData("f36"))).drop("f36")
      .withColumn("f57", toNum(chosenData("f37"))).drop("f37")
      .withColumn("f58", toNum(chosenData("f38"))).drop("f38")
      .withColumn("f59", toNum(chosenData("f39"))).drop("f39")
      .withColumn("f60", toNum(chosenData("f40"))).drop("f40")
      .withColumn("f61", toNum(chosenData("f41"))).drop("f41")
      .withColumn("f62", toNum(chosenData("f42"))).drop("f42")
      .withColumn("f63", toNum(chosenData("f43"))).drop("f43")
      .withColumn("f64", toNum(chosenData("f44"))).drop("f44")
      .withColumn("f65", toNum(chosenData("f45"))).drop("f45")
      .withColumn("f66", toNum(chosenData("f46"))).drop("f46")
      .withColumn("f67", toNum(chosenData("f47"))).drop("f47")
      .withColumn("f68", toNum(chosenData("f48"))).drop("f48")
      .withColumn("f69", toNum(chosenData("f49"))).drop("f49")
      .withColumn("f70", toNum(chosenData("f50"))).drop("f50")
      .na.fill(-1)

    numericDF.schema
    numericDF.show(20)

    // Assemble My Features
    val assembler = new VectorAssembler()
      .setInputCols(Array("f1",
        "f52",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
        "f13",
        "f14",
        "f15",
        "f16",
        "f17",
        "f18",
        "f19",
        "f20",
        "f21",
        "f22",
        "f23",
        "f24",
        "f25",
        "f26",
        "f27",
        "f28",
        "f29",
        "f30",
        "f31",
        "f32",
        "f53",
        "f54",
        "f55",
        "f56",
        "f57",
        "f58",
        "f59",
        "f60",
        "f61",
        "f62",
        "f63",
        "f64",
        "f65",
        "f66",
        "f67",
        "f68",
        "f69",
        "f70"))
      .setOutputCol("features")

    val featurizedDF = assembler.transform(numericDF)
      .select("features")
    featurizedDF.printSchema()
    featurizedDF.show()

    // Normalize the features
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)

    val normalizedDF = normalizer.transform(featurizedDF)
    normalizedDF.printSchema()
    normalizedDF.show()

    val kmeans = new KMeans()
      .setK(20)
      .setFeaturesCol("normFeatures")
      .setPredictionCol("clusterId")
    val model = kmeans.fit(normalizedDF)


    val predictedCluster = model.transform(normalizedDF)
    predictedCluster.printSchema()
    predictedCluster.show()


    session.stop()
  }
}
