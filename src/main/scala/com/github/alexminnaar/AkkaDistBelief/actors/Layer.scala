package com.github.alexminnaar.AkkaDistBelief.actors

import akka.actor.{Actor, ActorRef}
import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.alexminnaar.AkkaDistBelief.NeuralNetworkOps
import com.github.alexminnaar.AkkaDistBelief.NeuralNetworkOps._
import com.github.alexminnaar.AkkaDistBelief.Types.{Activation, ActivationFunction, Delta, LayerWeight}
import com.github.alexminnaar.AkkaDistBelief.actors.DataShard.{FetchParameters, ReadyToProcess}
import com.github.alexminnaar.AkkaDistBelief.actors.OutputActor.Output
import com.github.alexminnaar.AkkaDistBelief.actors.ParameterShard.{LatestParameters, ParameterRequest}

object Layer {

  case class DoneFetchingParameters(layerId: Int)

  case class Gradient(g: DenseMatrix[Double], replicaId: Int, layerId: Int)

  case class ForwardPass(inputs: DenseVector[Double], target: DenseVector[Double])

  case class BackwardPass(deltas: Delta)

  case class MyChild(ar: ActorRef)

}

/**
 * An Akka actor representing a layer in a DistBelief neural network replica.
 * @param replicaId A unique model replica identifier.
 * @param layerId The layer of the replica that this actor represents
 * @param activationFunction Neural network activation function.
 * @param activationFunctionDerivative Derivative of activation function.
 * @param parentLayer actorRef of this actor's parent layer (may not exist if this is an input layer).
 * @param parameterShardId actorRef of parameter shard corresponding to this layer.
 * @param outputAct actorRef of output actor.
 */
class Layer(replicaId: Int,
            layerId: Int,
            activationFunction: ActivationFunction,
            activationFunctionDerivative: ActivationFunction,
            parentLayer: Option[ActorRef],
            parameterShardId: ActorRef,
            outputAct: Option[ActorRef]) extends Actor {

  import com.github.alexminnaar.AkkaDistBelief.actors.Layer._


  var latestWeights: LayerWeight = _
  var activations: Activation = _
  var activatedInput: Activation = _
  var childLayer: Option[ActorRef] = None

  def receive = {

    //Before we process a data point, we must update the parameter weights for this layer
    case FetchParameters => {
      parameterShardId ! ParameterRequest(replicaId, layerId)
      context.become(waitForParameters)
    }

    //If this layer has a child, its identity will be sent.
    case MyChild(ar) => childLayer = Some(ar)

    case ForwardPass(inputs, target) => {

      activatedInput = parentLayer match {
        case Some(p) => DenseVector.vertcat(DenseVector(1.0), activationFunction(inputs))
        case _ => inputs
      }

      //compute outputs given current weights and received inputs, then pass them through activation function
      val outputs = computeLayerOutputs(activatedInput, latestWeights)
      val activatedOutputs = activationFunction(outputs)

      activations = parentLayer match {
        case Some(p) => DenseVector.vertcat(DenseVector(1.0), inputs)
        case _ => inputs
      }

      childLayer match {

        //If there is a child layer, send it the outputs with an added bias
        case Some(nextLayer) => {
          nextLayer ! ForwardPass(outputs, target)
        }

        //if this is the final layer of the neural network, compute prediction error and send the result backwards.
        case _ => {

          //compute deltas which we can use to compute the gradients for this layer's weights.
          val deltas = computePredictionError(activatedOutputs, target)
          val gradient = computeGradient(deltas, activatedInput)

          //send gradients for updating in the parameter shard actor
          parameterShardId ! Gradient(gradient, replicaId, layerId)

          //compute the deltas for this parent layer (there must be one if this is the output layer)
          val parentDeltas = computeDeltas(deltas, activations, latestWeights, activationFunctionDerivative)
          context.sender() ! BackwardPass(parentDeltas)

          //If this is the last layer then send the predictions to the output actor
          outputAct.get ! Output(replicaId, target, activatedOutputs)
        }
      }

    }

    case BackwardPass(childDeltas) => {

      //compute gradient of layer weights given deltas from child layer and activations from forward pass and
      //send the resulting gradient to the parameter shard for updating.
      val gradient = computeGradient(childDeltas, activatedInput)
      parameterShardId ! Gradient(gradient, replicaId, layerId)

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
