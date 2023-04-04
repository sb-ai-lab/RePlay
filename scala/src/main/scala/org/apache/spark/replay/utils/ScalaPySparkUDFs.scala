package org.apache.spark.replay.utils

import breeze.numerics.log2
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

object ScalaPySparkUDFs {

    private def _getMAPMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        val length = k.min(pred.length)
        if (groundTruth.length == 0 || pred.length == 0) {
            return 0
        }
        var tpCum = 0
        var result = 0.0
        for (i <- 0 until length) {
            if (groundTruth.contains(pred(i))) {
                tpCum += 1
                result = result + tpCum.toDouble / (i + 1)
            }
        }
        result / k
    }

    val getMAPMetricValue: UserDefinedFunction = udf(_getMAPMetricValue _)

    private def _getHitRateMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        val length = k.min(pred.length)
        for (i <- 0 until length) {
            if (groundTruth.contains(pred(i))) {
                return 1
            }
        }
        0
    }

    val getHitRateMetricValue: UserDefinedFunction = udf(_getHitRateMetricValue _)

    private def _getNDCGMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        if (pred.length == 0 || groundTruth.length == 0)
            return 0
        val length = k.min(pred.length)
        val groundTruthLen = k.min(groundTruth.length)
        val denom: Array[Double] = (0 until k map(x => 1.0/log2(x + 2))).toArray
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

    val getNDCGMetricValue: UserDefinedFunction = udf(_getNDCGMetricValue _)

    private def _getRocAucMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        val length = k.min(pred.length)
        if (groundTruth.length == 0 || pred.length == 0) {
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
        1.0 - fpCum.toDouble / (fpCur * (length - fpCur))
    }

    val getRocAucMetricValue: UserDefinedFunction = udf(_getRocAucMetricValue _)

    private def _getMRRMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        val length = k.min(pred.length)
        for (i <- 0 until length) {
            if (groundTruth.contains(pred(i))) {
                return 1.0 / (1.0 + i)
            }
        }
        0
    }

    val getMRRMetricValue: UserDefinedFunction = udf(_getMRRMetricValue _)

    private def _getPrecisionMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        if (pred.length == 0) {
            return 0
        }
        val length = k.min(pred.length)
        val i = pred.take(length).toSet.intersect(groundTruth.toSet).size
        i.toDouble / k
    }

    val getPrecisionMetricValue: UserDefinedFunction = udf(_getPrecisionMetricValue _)

    private def _getRecallMetricValue(k: Int, pred: Array[Int], groundTruth: Array[Int]) : Double = {
        if (groundTruth.length == 0) {
            return 0
        }
        val length = k.min(pred.length)
        val i = pred.take(length).toSet.intersect(groundTruth.toSet).size
        i.toDouble / groundTruth.length
    }

    val getRecallMetricValue: UserDefinedFunction = udf(_getRecallMetricValue _)

    def getSurprisalMetricValue: UserDefinedFunction = udf { (k: Int, weigths: Array[Double]) =>
        weigths.take(k).sum / k
    }

    private def _getUnexpectednessMetricValue(k: Int, pred: Array[Int], basePred: Array[Int]) : Double = {
        if (pred.length == 0) {
            return 0
        }
        1.0 - ( pred.take(k).toSet.intersect(basePred.take(k).toSet).size.toDouble / k )
    }

    val getUnexpectednessMetricValue: UserDefinedFunction = udf(_getUnexpectednessMetricValue _)

    private def _getNCISPrecisionMetricValue(k: Int, pred: Array[Int], predWeights: Array[Double], groundTruth: Array[Int]) : Double = {
        if (pred.length == 0 || predWeights.length == 0) {
            return 0
        }
        val length = k.min(pred.length)
        var sum1 = 0.0
        for (i <- 0 until length) {
            if (groundTruth.contains(pred(i))) {
                if (i < predWeights.length) {
                    sum1 += predWeights(i)
                }
            }
        }
        sum1 / predWeights.take(k).sum
    }

    val getNCISPrecisionMetricValue: UserDefinedFunction = udf(_getNCISPrecisionMetricValue _)

}
