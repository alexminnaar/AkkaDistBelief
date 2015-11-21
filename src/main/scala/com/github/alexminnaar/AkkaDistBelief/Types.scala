package com.github.alexminnaar.AkkaDistBelief

import breeze.linalg.{DenseMatrix, DenseVector}


object Types {

  type Activation = DenseVector[Double]
  type ActivationFunction = DenseVector[Double] => DenseVector[Double]
  type LayerWeight = DenseMatrix[Double]
  type Delta = DenseVector[Double]

}
