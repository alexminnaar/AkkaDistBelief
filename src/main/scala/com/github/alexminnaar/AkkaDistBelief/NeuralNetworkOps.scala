package com.github.alexminnaar.AkkaDistBelief

import breeze.linalg.{Axis, *, DenseMatrix, DenseVector}
import breeze.numerics._
import breeze.stats.distributions.Gaussian


object NeuralNetworkOps {

  //def sigmoidPrime(dv: DenseVector[Double]): DenseVector[Double] = dv.map(x => sigmoid(x) * ( - sigmoid(x)))

  def computeLayerOutputs(input: DenseVector[Double]
                          , weights: DenseMatrix[Double]): DenseVector[Double] = {

    weights * input
  }


  def computeDeltas(childDeltas: DenseVector[Double]
                    , thisLayerActivations: DenseVector[Double]
                    , currentWeights: DenseMatrix[Double]
                    , activationFunctionDerivative: DenseVector[Double] => DenseVector[Double]): DenseVector[Double] = {

    //println("delta comp child deltas ",childDeltas)
    //println("delta comp weights ",currentWeights.delete(0, Axis._1))
    //println("activations ",thisLayerActivations(1 to -1))

    val dw = currentWeights.delete(0, Axis._1).t * childDeltas
    //println("dw ", dw)
    //println("plain activations ",thisLayerActivations(1 to -1))
    //println("activation derivative, ", activationFunctionDerivative(thisLayerActivations(1 to -1)))

    activationFunctionDerivative(thisLayerActivations(1 to -1)) :* dw
  }


  def computeGradient(deltas: DenseVector[Double]
                      , thisLayerActivations: DenseVector[Double]): DenseMatrix[Double] = {

    println("deltas: ",deltas)
    println("activations ",thisLayerActivations)
    outerProd(deltas, thisLayerActivations)
  }


  def computePredictionError(prediction: DenseVector[Double]
                             , target: DenseVector[Double]): DenseVector[Double] = {

    prediction:* (1.0- prediction):*(target - prediction)
  }

  //Outer-product for two vectors
  def outerProd(v1: DenseVector[Double], v2: DenseVector[Double]): DenseMatrix[Double] = {

    var newV1: DenseMatrix[Double] = DenseMatrix(v1.toArray)

    while (newV1.rows != v2.length) {
      newV1 = DenseMatrix.vertcat(newV1, v1.toDenseMatrix)
    }

    val bc = newV1(::, *) *= v2
    bc.underlying
  }

  def randomMatrix(numRows: Int, numCols: Int): DenseMatrix[Double] = {

    val samples = Gaussian(0, 1).sample(numRows * numCols).toArray
    new DenseMatrix[Double](numRows, numCols, samples)
  }
}
