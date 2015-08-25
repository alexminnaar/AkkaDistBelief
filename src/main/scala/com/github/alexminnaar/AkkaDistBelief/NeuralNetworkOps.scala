package com.github.alexminnaar.AkkaDistBelief

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.stats.distributions.Gaussian


object NeuralNetworkOps {

  def computeLayerOutputs(input: DenseVector[Double]
                          , weights: DenseMatrix[Double]): DenseVector[Double] = {
    weights * input
  }


  def computeDeltas(childDeltas: DenseVector[Double]
                    , thisLayerActivations: DenseVector[Double]
                    , currentWeights: DenseMatrix[Double]
                    , activationFunctionDerivative: DenseVector[Double] => DenseVector[Double]): DenseVector[Double] = {

    val dw = currentWeights.t * childDeltas
    activationFunctionDerivative(thisLayerActivations) :* dw
  }


  def computeGradient(deltas: DenseVector[Double]
                      , thisLayerActivations: DenseVector[Double]
                      , activationFunction: DenseVector[Double] => DenseVector[Double]): DenseMatrix[Double] = {
    outerProd(deltas, activationFunction(thisLayerActivations))
  }


  def computePredictionError(prediction: DenseVector[Double], target: DenseVector[Double]): DenseVector[Double] = {
    target - prediction
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
