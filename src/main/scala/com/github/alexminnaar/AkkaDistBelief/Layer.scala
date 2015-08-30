package com.github.alexminnaar.AkkaDistBelief

import akka.actor.{ActorRef, Actor}
import breeze.linalg.{Axis, *, DenseVector, DenseMatrix}
import com.github.alexminnaar.AkkaDistBelief.DataShard.{ReadyToProcess, FetchParameters}
import com.github.alexminnaar.AkkaDistBelief.ParameterShard.{ParameterRequest, LatestParameters}
import NeuralNetworkOps._

object Layer {

  case class DoneFetchingParameters(layerId: Int)

  case class Gradient(g: DenseMatrix[Double])

  case class ForwardPass(inputs: DenseVector[Double], target: DenseVector[Double])

  case class BackwardPass(deltas: DenseVector[Double])

  case class MyChild(ar:ActorRef)

}

class Layer(replicaId: Int
            , layerId: Int
            , activationFunction: DenseVector[Double] => DenseVector[Double]
            , activationFunctionDerivative: DenseVector[Double] => DenseVector[Double]
            , parentLayer: Option[ActorRef]
            , parameterShardId: ActorRef
             ,outputAct:Option[ActorRef]) extends Actor {

  import com.github.alexminnaar.AkkaDistBelief.Layer._
  import com.github.alexminnaar.AkkaDistBelief.OutputActor.Output


  var latestWeights: DenseMatrix[Double] = _
  var activations: DenseVector[Double] = _
  var childLayer:Option[ActorRef]= None

  def receive = {

    //Before we process a data point, we must update the parameter weights for this layer
    case FetchParameters => {
      parameterShardId ! ParameterRequest
      context.become(waitForParameters)
    }

    //If this layer has a child, its identity will be sent.
    case MyChild(ar) => childLayer=Some(ar)

    case ForwardPass(inputs, target) => {

      //compute outputs given current weights and received inputs, then pass them through activation function
      val outputs = computeLayerOutputs(inputs, latestWeights)
      val activatedOutputs = activationFunction(outputs)
      val outputWithBias = DenseVector.vertcat(DenseVector(1.0), activatedOutputs)
      activations = inputs

      childLayer match {

        //If there is a child layer, send it the outputs with an added bias
        case Some(nextLayer) => {
          nextLayer ! ForwardPass(outputWithBias, target)
        }

        //if this is the final layer of the neural network, compute prediction error and send the result backwards.
        case _ => {

          //compute deltas which we can use to compute the gradients for this layer's weights.
          val deltas=computePredictionError(activatedOutputs, target)
          val gradient = computeGradient(deltas, activations, activationFunction)

          //send gradients for updating in the parameter shard actor
          parameterShardId ! Gradient(gradient)

          //compute the deltas for this parent layer (there must be one if this is the output layer)
          val parentDeltas = computeDeltas(deltas, activations, latestWeights, activationFunctionDerivative)
          context.sender() ! BackwardPass(parentDeltas)

          //If this is the last layer then send the predictions to the output actor
          outputAct.get ! Output(replicaId,activatedOutputs)
        }
      }

    }

    case BackwardPass(childDeltas) => {

      //compute gradient of layer weights given deltas from child layer and activations from forward pass and
      //send the resulting gradient to the parameter shard for updating.
      val gradient = computeGradient(childDeltas, activations, activationFunction)
      parameterShardId ! Gradient(gradient)

      parentLayer match {

        //If there is a parent layer, compute deltas for this layer and send them backwards.  We remove the delta
        //corresponding to the bias unit because it is not connected to anything in the parent layer thus it should
        //not affect its gradient.
        case Some(previousLayer) => {
          val parentDeltas = computeDeltas(childDeltas, activations, latestWeights, activationFunctionDerivative)
          previousLayer ! BackwardPass(parentDeltas(1 to -1))
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


}
