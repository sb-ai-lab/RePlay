package org.apache.spark.replay.utils

import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.Vectors
import breeze.numerics.log2

object ScalaPySparkUDFs {

    def multiplyUDF = udf { (scalar: Double, vector: DenseVector) =>
        val resultVector = new Array[Double](vector.size)
        vector.foreachActive {(index: Int, value: Double) => 
            resultVector(index) = scalar*value
        }
        Vectors.dense(resultVector)
    }

}
