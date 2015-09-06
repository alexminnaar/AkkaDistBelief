package com.github.alexminnaar.AkkaDistBelief.actors

import akka.actor.{ActorLogging, Actor}
import breeze.linalg.DenseVector

object OutputActor {

  case class Output(replicaId: Int, target: DenseVector[Double], output: DenseVector[Double])

}

/**
 * Actor that logs outputs and keeps track of the last predictions of each model replica
 */
class OutputActor extends Actor with ActorLogging{

  import com.github.alexminnaar.AkkaDistBelief.actors.OutputActor._

  var latestOutputs: Map[Int, DenseVector[Double]] = Map.empty

  def receive = {

    case Output(replica, target, output) => {

      latestOutputs += (replica -> output)

      log.info(s"replica id ${replica}, output: ${output}, target ${target}")
    }


  }


}
