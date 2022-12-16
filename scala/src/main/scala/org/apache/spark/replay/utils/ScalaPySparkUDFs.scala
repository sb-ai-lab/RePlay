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

    def _getMAPMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        val length = k.min(pred.size)
        val maxGood = k.min(groundTruth.size)
        if (groundTruth.size == 0 || pred.size == 0) {
            return 0
        }
        var tpCum = 0
        var result = 0.0
        for (i <- 0 until length) {
            if (groundTruth.contains(pred(i))) {
                tpCum += 1
                result = result + tpCum.toDouble / ((i + 1) * maxGood)
            }
        }
        return result
    }

    val getMAPMetricValue = udf(_getMAPMetricValue _)

    def _getHitRateMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        val length = k.min(pred.size)
        for (i <- 0 until length) {
            if (groundTruth.contains(pred(i))) {
                return 1
            }
        }
        return 0
    }

    val getHitRateMetricValue = udf(_getHitRateMetricValue _)

    def getNDCGMetricValue = udf { (k: Int, pred: Array[Int], groundTruth: Array[Int]) =>
        val length = k.min(pred.size)
        val groundTruthLen = k.min(groundTruth.size)
        val denom:Array[Double] = (0 until k map(x => 1.0/log2(x + 2))).toArray
        var dcg = 0.0
        for (i <- 0 until length) {
            if (groundTruth.contains(pred(i))) {
                dcg += denom(i)
            }
        }
        var idcg = 0.0
        for (i <- 0 until groundTruthLen) {
            idcg += denom(i)
        }
        dcg / idcg    
    }

    def _getRocAucMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        val length = k.min(pred.size)
        if (groundTruth.size == 0 || pred.size == 0) {
            return 0
        }
        var fpCur = 0
        var fpCum = 0
        for (i <- 0 until length) {
            if (groundTruth.contains(pred(i))) {
                fpCum += fpCur
            } else {
                fpCur += 1
            }
        }
        if (fpCur == length) {
            return 0
        }
        if (fpCum == 0) {
            return 1
        }
        return 1.0 - fpCum.toDouble / (fpCur * (length - fpCur))
    }

    val getRocAucMetricValue = udf(_getRocAucMetricValue _)

    def _getMRRMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        val length = k.min(pred.size)
        for (i <- 0 until length) {
            if (groundTruth.contains(pred(i))) {
                return 1.0 / (1.0 + i)
            }
        }
        return 0
    }

    val getMRRMetricValue = udf(_getMRRMetricValue _)

    def _getPrecisionMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        if (pred.size == 0) {
            return 0
        }
        val length = k.min(pred.size)
        val i = pred.take(length).toSet.intersect(groundTruth.toSet).size
        return i.toDouble / length
    }

    val getPrecisionMetricValue = udf(_getPrecisionMetricValue _)

    def _getRecallMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        val length = k.min(pred.size)
        val i = pred.take(length).toSet.intersect(groundTruth.toSet).size
        return i.toDouble / groundTruth.size
    }

    val getRecallMetricValue = udf(_getRecallMetricValue _)

    def getSurprisalMetricValue = udf { (k: Int, weigths: Array[Double]) =>
        weigths.take(k).sum / k
    }

    def _getUnexpectednessMetricValue(k: Int, pred: Array[Int], basePred: Array[Int]) : Double = {
        if (pred.size == 0) {
            return 0
        }
        return 1.0 - ( pred.take(k).toSet.intersect(basePred.take(k).toSet).size.toDouble / k )
    }

    val getUnexpectednessMetricValue = udf(_getUnexpectednessMetricValue _)

    def _getNCISPrecisionMetricValue(k: Int, pred: Array[Int], predWeights: Array[Double], groundTruth: Array[Int]) : Double = {
        if (pred.size == 0) {
            return 0
        }
        val length = k.min(pred.size)
        var sum1 = 0.0
        for (i <- 0 until length) {
            if (groundTruth.contains(pred(i))) {
                if (i < predWeights.size) {
                    sum1 += predWeights(i)
                }
            }
        }
        return sum1 / predWeights.take(k).sum
    }

    val getNCISPrecisionMetricValue = udf(_getNCISPrecisionMetricValue _)

}
