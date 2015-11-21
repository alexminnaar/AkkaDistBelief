package com.github.alexminnaar.AkkaDistBelief

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector}
import breeze.stats.distributions.Gaussian
import com.github.alexminnaar.AkkaDistBelief.Types.{Activation, ActivationFunction, Delta, LayerWeight}

/*
A collection of neural network operations required for backpropagation
 */
object NeuralNetworkOps {

  /**
   * Compute the weighted sum of a neural network layer.
   * @param input Input to layer.
   * @param weights Weight parameters associated with layer.
   * @return Weighted sum of input.
   */
  def computeLayerOutputs(input: Activation,
                          weights: LayerWeight): Activation = {
    weights * input
  }

  /**
   * Compute the deltas for a neural network layer.
   * @param childDeltas deltas of child layer.
   * @param thisLayerActivations activations of current layer.
   * @param currentWeights weights associated with layer.
   * @param activationFunctionDerivative derivative of activation function for this layer.
   * @return This layer's deltas.
   */
  def computeDeltas(childDeltas: Delta,
                    thisLayerActivations: Activation,
                    currentWeights: LayerWeight,
                    activationFunctionDerivative: ActivationFunction): Delta = {

    val dw = currentWeights.delete(0, Axis._1).t * childDeltas
    activationFunctionDerivative(thisLayerActivations(1 to -1)) :* dw
  }

  /**
   * Compute gradients for a a set of weight parameters.
   * @param deltas delta's associated with child layer.
   * @param thisLayerActivations activations of current layer.
   * @return weight gradients.
   */
  def computeGradient(deltas: Delta,
                      thisLayerActivations: Activation): DenseMatrix[Double] = {

    outerProd(deltas, thisLayerActivations)
  }

  /**
   * Compute prediction error for a prediction and target
   */
  def computePredictionError(prediction: DenseVector[Double]
                             , target: DenseVector[Double]): DenseVector[Double] = {

    prediction :* (1.0 - prediction) :* (target - prediction)
  }

  /**
   * Outer-product for two vectors
   */
  def outerProd(v1: DenseVector[Double], v2: DenseVector[Double]): DenseMatrix[Double] = {

    var newV1: DenseMatrix[Double] = DenseMatrix(v1.toArray)

    while (newV1.rows != v2.length) {
      newV1 = DenseMatrix.vertcat(newV1, v1.toDenseMatrix)
    }

    val bc = newV1(::, *) *= v2
    bc.underlying
  }

  /**
   * Create a random DenseMatrix from samples from Normal(0,0.2) distribution
   */
  def randomMatrix(numRows: Int, numCols: Int): LayerWeight = {

    val samples = Gaussian(0, 0.2).sample(numRows * numCols).toArray
    new DenseMatrix[Double](numRows, numCols, samples)
  }
}
