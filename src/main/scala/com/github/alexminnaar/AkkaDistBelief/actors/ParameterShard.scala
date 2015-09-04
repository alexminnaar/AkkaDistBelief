package com.github.alexminnaar.AkkaDistBelief.actors

import akka.actor.Actor
import breeze.linalg.DenseMatrix
import com.github.alexminnaar.AkkaDistBelief.actors.Layer.Gradient


object ParameterShard {

  case object ParameterRequest

  case class LatestParameters(weights: DenseMatrix[Double])

}

class ParameterShard(shardId: Int
                     , learningRate: Double
                     , initialWeight: DenseMatrix[Double]) extends Actor {

  //initialize randomly
  var latestParameter: DenseMatrix[Double] = initialWeight

  def receive = {

    //A layer corresponding to this shardId in some model replica has requested the latest version of the parameters.
    case ParameterRequest => context.sender() ! LatestParameters(latestParameter)

    /*
    A layer corresponding to this shardId in some model replica has computed a gradient, so we must update our
    parameters according to this gradient.
    */
    case Gradient(g) => {

      latestParameter = latestParameter + g.t*learningRate

    }

  }

}
