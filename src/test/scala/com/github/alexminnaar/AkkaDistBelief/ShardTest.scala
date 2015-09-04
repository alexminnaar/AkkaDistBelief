package com.github.alexminnaar.AkkaDistBelief

import akka.actor.ActorSystem
import akka.testkit.{TestActorRef, TestKit}
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._
import com.github.alexminnaar.AkkaDistBelief.actors.{ParameterShard, Master, Layer, DataShard}
import Layer.Gradient
import ParameterShard.ParameterRequest
import org.scalatest.{MustMatchers, WordSpecLike}


class ShardTest extends TestKit(ActorSystem("testSystem")) with WordSpecLike with MustMatchers {

  val dataset = Seq(

    //For shard 1
    Example(DenseVector(1.0, 0.0, 0.0), DenseVector(0.0))
    , Example(DenseVector(1.0, 0.0, 1.0), DenseVector(1.0))
    , Example(DenseVector(1.0, 1.0, 0.0), DenseVector(1.0))
    , Example(DenseVector(1.0, 1.0, 1.0), DenseVector(0.0))

    //For shard 2
    , Example(DenseVector(1.0, 0.0, 0.0), DenseVector(0.0))
    , Example(DenseVector(1.0, 0.0, 1.0), DenseVector(1.0))
    , Example(DenseVector(1.0, 1.0, 0.0), DenseVector(1.0))
    , Example(DenseVector(1.0, 1.0, 1.0), DenseVector(0.0))

    //For shard 3
    , Example(DenseVector(1.0, 0.0, 0.0), DenseVector(0.0))
    , Example(DenseVector(1.0, 0.0, 1.0), DenseVector(1.0))
    , Example(DenseVector(1.0, 1.0, 0.0), DenseVector(1.0))
    , Example(DenseVector(1.0, 1.0, 1.0), DenseVector(0.0))
  )

  "Master" should {

    "create the correct data partitions" in {

      val masterTestActor = TestActorRef(new Master(
        dataset
        , 4
        , Seq(3, 3, 1)
        , (x: DenseVector[Double]) => x.map(el => sigmoid(el))
        , (x: DenseVector[Double]) => x.map(el => sigmoid(el) * (1 - sigmoid(el)))
        , 0.5
      ))


      val dataPartition = masterTestActor.underlyingActor.dataShards

      dataPartition must equal(Seq(
        Seq(Example(DenseVector(1.0, 0.0, 0.0), DenseVector(0.0))
          , Example(DenseVector(1.0, 0.0, 1.0), DenseVector(1.0))
          , Example(DenseVector(1.0, 1.0, 0.0), DenseVector(1.0))
          , Example(DenseVector(1.0, 1.0, 1.0), DenseVector(0.0)))
        , Seq(Example(DenseVector(1.0, 0.0, 0.0), DenseVector(0.0))
          , Example(DenseVector(1.0, 0.0, 1.0), DenseVector(1.0))
          , Example(DenseVector(1.0, 1.0, 0.0), DenseVector(1.0))
          , Example(DenseVector(1.0, 1.0, 1.0), DenseVector(0.0)))
        , Seq(Example(DenseVector(1.0, 0.0, 0.0), DenseVector(0.0))
          , Example(DenseVector(1.0, 0.0, 1.0), DenseVector(1.0))
          , Example(DenseVector(1.0, 1.0, 0.0), DenseVector(1.0))
          , Example(DenseVector(1.0, 1.0, 1.0), DenseVector(0.0)))

      ))

    }

  }

  "ParameterShard" should {

    val parameterShardTestActor = TestActorRef(new ParameterShard(
      0
      , 0.5
      , DenseMatrix((0.2, 0.4), (0.5, 0.2))
    ))

    "return correct parameters when requested" in {

      parameterShardTestActor ! ParameterRequest

      parameterShardTestActor.underlyingActor.latestParameter must equal(DenseMatrix((0.2, 0.4), (0.5, 0.2)))

    }

    "update parameters correctly when sent a new gradient" in {

      val g = DenseMatrix((0.1, 0.1), (0.1, 0.1))

      parameterShardTestActor ! Gradient(g)

      parameterShardTestActor.underlyingActor.latestParameter must equal(DenseMatrix((0.25, 0.45), (0.55, 0.25)))

    }
  }


  "DataShard" should {

    val parameterShardTestActor1 = TestActorRef(new ParameterShard(
      1,
      0.1,
      DenseMatrix((0.341232, 0.129952, -0.923123)
        , (-0.115223, 0.570345, -0.328932))))

    val parameterShardTestActor2 = TestActorRef(new ParameterShard(
      1,
      0.1,
      DenseMatrix((-0.993423, 0.164732, 0.752621))))

    val dataShardTestActor = TestActorRef(new DataShard(
      0
      , dataset
      , (x: DenseVector[Double]) => x.map(el => sigmoid(el))
      , (x: DenseVector[Double]) => x.map(el => sigmoid(el) * (1 - sigmoid(el)))
      , Seq(parameterShardTestActor1, parameterShardTestActor2)
    ))

    "have the correct number of layers not updated" in {

      dataShardTestActor.underlyingActor.layersNotUpdated must equal(Set(0, 1))

    }

  }


}
