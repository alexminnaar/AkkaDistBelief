package com.github.alexminnaar.AkkaDistBelief

import akka.actor.Actor
import breeze.linalg.DenseVector

object OutputActor {

  case class Output(replicaId: Int, output: DenseVector[Double])

}

class OutputActor extends Actor {

  import OutputActor._


  def receive = {

    case Output(replica, output) => println(s"replica id ${replica}, output: ${output}")


  }


}
