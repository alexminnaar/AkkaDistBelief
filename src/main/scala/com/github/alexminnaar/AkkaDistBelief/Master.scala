package com.github.alexminnaar.AkkaDistBelief

import akka.actor.{ActorRef, Props, Actor}
import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Gaussian
import com.github.alexminnaar.AkkaDistBelief.DataShard.ReadyToProcess
import com.github.alexminnaar.AkkaDistBelief.Master.Start

object Master {


  case class Done(dataShardId: Int)

  case object Start

}


class Master(dataSet: Seq[Example],
             numReplicas: Int,
             layerDimensions: Seq[Int],
             learningRate: Double) extends Actor {

  val numLayers = layerDimensions.size

  //split dataset into shards
  val dataShards = dataSet.grouped(numReplicas).toSeq


  //create parameter shards for each layer
  val parameterShardActors: Array[ActorRef] = new Array[ActorRef](numLayers - 1)

  for (i <- 0 to numLayers - 2) {

    parameterShardActors(i) = context.actorOf(Props(new ParameterShard(
      i
      , learningRate
      , randomMatrix(layerDimensions(i), layerDimensions(i + 1))
    )))
  }


  //create actors for each data shard/replica.  Each replica needs to know about all parameter shards because they will
  //be reading from them and updating them
  val dataShardActors = dataShards.zipWithIndex.map { dataShard =>
    context.actorOf(Props(new DataShard(dataShard._2, dataShard._1, parameterShardActors)))
  }


  def receive = {

    case Start => dataShardActors.foreach(_ ! ReadyToProcess)

  }


  def randomMatrix(numRows: Int, numCols: Int): DenseMatrix[Double] = {

    val samples = Gaussian(0, 1).sample(numRows * numCols).toArray
    new DenseMatrix[Double](numRows, numCols, samples)
  }
}
