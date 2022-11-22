package org.apache.spark.replay.utils

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.SparkSession
import org.apache.spark.replay.utils.ScalaPySparkUDFs.multiplyUDF
import org.apache.spark.ml.linalg.Vectors



object TestUDF extends App {

  val spark = SparkSession
          .builder()
          .appName("test")
          .master("local[1]")
          .getOrCreate()

    import spark.implicits._

    val df = Seq(
        (8, Vectors.dense(1.0, 0.0, 2.0)),
        (64, Vectors.dense(1.0, 0.0, 2.0)),
        (-27, Vectors.dense(1.0, 0.0, 2.0))
        ).toDF("scalar", "vector")
    df.show()

    df.select(multiplyUDF(col("scalar"), col("vector")).alias("product")).show()

}
