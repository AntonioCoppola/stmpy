/*
 * Copyright 2015 by Antonio Coppola, Harvard University.
 * Released under the MIT license for non-commercial use.
 */

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.Logging
import org.apache.spark.annotation.{DeveloperApi, Experimental, Since}
import org.apache.spark.api.java.JavaPairRDD
import org.apache.spark.graphx._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils

/**
 * :: Experimental ::
 *
 * Structural Topic Model (STM), a covariate-augmented topic model.
 *
 * Terminology:
 *  - "word" = "term": an element of the vocabulary
 *  - "token": instance of a term appearing in a document
 *  - "topic": multinomial distribution over words representing some concept
 */
@Since("1.3.0")
@Experimental
class STM private (
    private var k: Int,
    private var maxIterations: Int,
    private var docConcentration: Vector,
    private var topicConcentration: Double,
    private var seed: Long,
    private var checkpointInterval: Int,
    private var optimizer: STMOptimizer) extends Logging {
  
}
