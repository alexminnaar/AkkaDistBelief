package com.github.alexminnaar.AkkaDistBelief

import akka.actor.Actor
import breeze.linalg.{Axis, DenseMatrix}
import com.github.alexminnaar.AkkaDistBelief.Layer.Gradient


object ParameterShard {

  case object ParameterRequest

  case class LatestParameters(weights: DenseMatrix[Double])

}

class ParameterShard(shardId: Int
                     , learningRate: Double
                     , initialWeight: DenseMatrix[Double]) extends Actor {

  import com.github.alexminnaar.AkkaDistBelief.ParameterShard._

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

      println(s"gradient: ${g.t}")
      println(s"latestParameter: ${latestParameter}")

      val gtrans=g.t

      println(gtrans.rows,gtrans.cols)
      println(latestParameter.rows,latestParameter.cols)

      //val t=latestParameter + gtrans.delete(0,Axis._1)

      latestParameter = latestParameter + gtrans

      println("new weights: ",latestParameter)

    }

  }

}
