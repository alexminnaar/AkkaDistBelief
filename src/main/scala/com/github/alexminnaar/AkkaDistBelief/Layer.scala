package com.github.alexminnaar.AkkaDistBelief

import akka.actor.{ActorRef, Actor}
import breeze.linalg.{Axis, *, DenseVector, DenseMatrix}
import com.github.alexminnaar.AkkaDistBelief.DataShard.{ReadyToProcess, FetchParameters}
import com.github.alexminnaar.AkkaDistBelief.ParameterShard.{ParameterRequest, LatestParameters}

object Layer {


  case class DoneFetchingParameters(layerId: Int)

  case class Gradient(g: DenseMatrix[Double])

  case class ForwardPass(inputs: DenseVector[Double], target: DenseVector[Double])

  case class BackwardPass(deltas: DenseVector[Double])

}

class Layer(replicaId: Int
            , layerId: Int
            , activationFunction: DenseVector[Double] => DenseVector[Double]
            , activationFunctionDerivative: DenseVector[Double] => DenseVector[Double]
            , childLayer: Option[ActorRef]
            , parentLayer: Option[ActorRef]
            , parameterShardId: ActorRef) extends Actor {

  import com.github.alexminnaar.AkkaDistBelief.Layer._


  var latestWeights: DenseMatrix[Double] = _

  //we need to also keep track of the activations for this layer for the backwards pass.
  var activations: DenseVector[Double] = _

  def receive = {

    //Before we process a data point, we must update the parameter weights for this layer
    case FetchParameters => {
      parameterShardId ! ParameterRequest
      context.become(waitForParameters)
    }

    case ForwardPass(inputs, target) => {

      //compute outputs given current weights and received inputs, then pass them through activation function
      val outputs = computeLayerOutputs(inputs, latestWeights)
      val activatedOutputs = activationFunction(outputs)

      childLayer match {

        //If there is a child layer, send it the outputs with an added bias
        case Some(nextLayer) => {
          val outputWithBias = DenseVector.vertcat(DenseVector(1.0), activatedOutputs)
          nextLayer ! ForwardPass(outputWithBias, target)
          activations = activatedOutputs //also persist them for the backwards pass
        }

        //if this is the final layer of the neural network, compute prediction error and send the result backwards.
        case _ => context.sender() ! BackwardPass(computePredictionError(activatedOutputs, target))
      }

    }

    case BackwardPass(childDeltas) => {

      //compute gradient of layer weights given deltas from child layer and activations from forward pass and
      //send the resulting gradient to the parameter shard for updating.
      val gradient = computeGradient(childDeltas, activations)
      parameterShardId ! Gradient(gradient)

      parentLayer match {

        //If there is a parent layer, compute deltas for this layer and send them backwards.  We remove the delta
        //corresponding to the bias unit because it is not connected to anything in the parent layer thus it should
        //not affect its gradient.
        case Some(previousLayer) => {
          val deltas = computeDeltas(childDeltas, activations, latestWeights)
          previousLayer ! BackwardPass(deltas(1 to -1))
        }

        //If this is the first layer, let data shard know we are ready to update weights and process another data point.
        case _ => context.parent ! ReadyToProcess
      }

    }

  }


  def waitForParameters: Receive = {

    /*
    latest parameter update has been fetched from parameter server.  Send message to data shard (i.e. this actor's
    parent) indicating that this layer is now ready to process the next data point.
    */
    case LatestParameters(weights) => {
      latestWeights = weights
      context.parent ! DoneFetchingParameters(layerId)

      context.unbecome()
    }

  }

  //
  def computeLayerOutputs(input: DenseVector[Double]
                          , weights: DenseMatrix[Double]): DenseVector[Double] = {
    weights * input
  }


  def computeDeltas(childDeltas: DenseVector[Double]
                    , thisLayerActivations: DenseVector[Double]
                    , currentWeights: DenseMatrix[Double]): DenseVector[Double] = {

    val dw = currentWeights.t * childDeltas
    activationFunctionDerivative(thisLayerActivations) :* dw
  }


  def computeGradient(deltas: DenseVector[Double]
                      , thisLayerActivations: DenseVector[Double]): DenseMatrix[Double] = {
    outerProd(deltas, activationFunction(thisLayerActivations))
  }


  def computePredictionError(prediction: DenseVector[Double], target: DenseVector[Double]): DenseVector[Double] = {
    target - prediction
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
}
