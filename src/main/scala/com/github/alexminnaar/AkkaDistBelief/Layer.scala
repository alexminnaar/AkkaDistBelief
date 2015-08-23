package com.github.alexminnaar.AkkaDistBelief

import akka.actor.{ActorRef, Actor}
import breeze.linalg.{*, DenseVector, DenseMatrix}
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

    /*
    Before we process a data point, we must update the parameter weights for this layer
     */
    case FetchParameters => {
      parameterShardId ! ParameterRequest
      context.become(waitForParameters)
    }

    case ForwardPass(inputs, target) => {

      val outputs = computeLayerOutputs(inputs, latestWeights) //compute outputs given current weights
      val activatedOutputs=activationFunction(outputs)

      /*
      If this is not the final layer, compute outputs and send them to child layer.
      if this is the final layer of the neural network, compute prediction error and send the result backwards.
      */
      childLayer match {

        case Some(nextLayer) => {
          nextLayer ! ForwardPass(activatedOutputs, target) //send them to next layer
          activations = outputs //also persist them for the backwards pass
        }

        case _ => context.sender() ! BackwardPass(computePredictionError(activatedOutputs, target))
      }

    }

    case BackwardPass(childDeltas) => {

      val deltas = computeDeltas(childDeltas, activations, latestWeights)
      val gradient = computeGradient(deltas, activations)

      /*
      If there is a parent layer, send the gradients backwards.
      If this is the first layer, tell the data shard parent that it is ready to process another data point.
      */
      parentLayer match {
        case Some(previousLayer) => previousLayer ! BackwardPass(deltas)
        case _ => context.parent ! ReadyToProcess
      }

      //In either case we still need to send the gradient to the parameter shard for updating
      parameterShardId ! Gradient(gradient)
    }

  }


  def waitForParameters: Receive = {

    /*
    latest parameter update has been fetched from parameter server.  Send message to data shard (i.e. this actor's
    parent) indicating that this layer is now ready to process the next minibatch of data.
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
