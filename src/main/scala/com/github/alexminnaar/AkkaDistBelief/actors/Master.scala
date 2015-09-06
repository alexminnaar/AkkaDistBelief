package com.github.alexminnaar.AkkaDistBelief.actors

import akka.actor.{ActorLogging, Actor, ActorRef, Props}
import breeze.linalg.DenseVector
import com.github.alexminnaar.AkkaDistBelief.actors.DataShard.ReadyToProcess
import com.github.alexminnaar.AkkaDistBelief.{NeuralNetworkOps, Example}

object Master {

  case class Done(dataShardId: Int)

  case object Start

  case object JobDone

}

/**
 * The master actor of the DistBelief implementation.
 * @param dataSet The data set to be used for training.
 * @param dataPerReplica The number of data points to be used in a data shard.
 * @param layerDimensions The number of neural units in each layer of the neural network model.
 * @param activation The activation function.
 * @param activationDerivative The derivative of the activation function.
 * @param learningRate The learning rate for parameter updates.
 */
class Master(dataSet: Seq[Example],
             dataPerReplica: Int,
             layerDimensions: Seq[Int],
             activation: DenseVector[Double] => DenseVector[Double],
             activationDerivative: DenseVector[Double] => DenseVector[Double],
             learningRate: Double) extends Actor with ActorLogging{

  import com.github.alexminnaar.AkkaDistBelief.actors.Master._

  val numLayers = layerDimensions.size

  //split dataset into shards
  val dataShards = dataSet.grouped(dataPerReplica).toSeq


  //create parameter shards for each layer
  val parameterShardActors: Array[ActorRef] = new Array[ActorRef](numLayers - 1)

  for (i <- 0 to numLayers - 2) {

    parameterShardActors(i) = context.actorOf(Props(new ParameterShard(
      i
      , learningRate
      , NeuralNetworkOps.randomMatrix(layerDimensions(i+1), layerDimensions(i)+1)
    )))
  }

  log.info(s"${numLayers - 1} parameter shards initiated!")

  //create actors for each data shard/replica.  Each replica needs to know about all parameter shards because they will
  //be reading from them and updating them
  val dataShardActors = dataShards.zipWithIndex.map { dataShard =>
    context.actorOf(Props(new DataShard(dataShard._2
      , dataShard._1
      , activation
      , activationDerivative
      , parameterShardActors)))
  }

  log.info(s"${dataShards.size} data shards initiated!")

  var numShardsFinished = 0

  def receive = {

    case Start => dataShardActors.foreach(_ ! ReadyToProcess)

    case Done(id) => {

      numShardsFinished += 1

      log.info(s" ${ numShardsFinished} shards finished of ${dataShards.size}")

      if (numShardsFinished == dataShards.size) {
        context.parent ! JobDone
        context.stop(self)
      }

    }

  }


}
