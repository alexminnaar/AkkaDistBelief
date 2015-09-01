package com.github.alexminnaar.AkkaDistBelief

import akka.actor.{ActorRef, Props, Actor}
import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.distributions.Gaussian
import com.github.alexminnaar.AkkaDistBelief.DataShard.ReadyToProcess
import com.github.alexminnaar.AkkaDistBelief.Master.{Done, Start}

object Master {

  case class Done(dataShardId: Int)

  case object Start

}


class Master(dataSet: Seq[Example],
             dataPerReplica: Int,
             layerDimensions: Seq[Int],
             activation: DenseVector[Double] => DenseVector[Double],
             activationDerivative: DenseVector[Double] => DenseVector[Double],
             learningRate: Double) extends Actor {

  val numLayers = layerDimensions.size

  //split dataset into shards
  val dataShards = dataSet.grouped(dataPerReplica).toSeq


  //create parameter shards for each layer
  val parameterShardActors: Array[ActorRef] = new Array[ActorRef](numLayers - 1)

  for (i <- 0 to numLayers - 2) {

    parameterShardActors(i) = context.actorOf(Props(new ParameterShard(
      i
      , learningRate
      , NeuralNetworkOps.randomMatrix(layerDimensions(i), layerDimensions(i + 1))
    )))
  }

  //create actors for each data shard/replica.  Each replica needs to know about all parameter shards because they will
  //be reading from them and updating them
  val dataShardActors = dataShards.zipWithIndex.map { dataShard =>
    context.actorOf(Props(new DataShard(dataShard._2
      , dataShard._1
      , activation: DenseVector[Double] => DenseVector[Double]
      , activationDerivative: DenseVector[Double] => DenseVector[Double]
      , parameterShardActors)))
  }

  var numShardsFinished = 0

  def receive = {

    case Start => dataShardActors.foreach(_ ! ReadyToProcess)

    case Done(id) => {
      numShardsFinished += 1
      if (numShardsFinished == dataShards.size) println("DONE!!!!!!!!!!!!!!!")
    }

  }


}
