package com.github.alexminnaar.AkkaDistBelief.actors

import akka.actor.Actor
import breeze.linalg.DenseVector

object OutputActor {

  case class Output(replicaId: Int, output: DenseVector[Double])

}

class OutputActor extends Actor {

  import com.github.alexminnaar.AkkaDistBelief.actors.OutputActor._

  var latestOutputs: Map[Int, DenseVector[Double]] = Map.empty

  def receive = {

    case Output(replica, output) => {

      latestOutputs += (replica -> output)

      println(s"replica id ${replica}, output: ${output}")
    }


  }


}
