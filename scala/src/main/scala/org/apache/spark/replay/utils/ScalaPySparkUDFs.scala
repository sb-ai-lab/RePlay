package org.apache.spark.replay.utils

import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.UserDefinedFunction

object ScalaPySparkUDFs {

    def multiplyUDF: UserDefinedFunction = udf { (scalar: Double, vector: DenseVector) =>
        val resultVector = new Array[Double](vector.size)
        vector.foreachActive {(index: Int, value: Double) => 
            resultVector(index) = scalar*value
        }
        Vectors.dense(resultVector)
    }

}
