import akka.actor.ActorSystem
import akka.testkit.{TestActorRef, TestKit}
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import com.github.alexminnaar.AkkaDistBelief.DataShard.FetchParameters
import com.github.alexminnaar.AkkaDistBelief.{ParameterShard, Layer}
import org.scalatest.{WordSpecLike, MustMatchers}


class LayerTest extends TestKit(ActorSystem("testSystem")) with WordSpecLike with MustMatchers {

  "Layer actor" must {

    "fetch the correct parameters from its parameter shard" in {

      val testWeights = DenseMatrix((0.1, 0.2), (0.3, 0.4))

      //parameter shard associate with this layer
      val parameterShardTestActor = TestActorRef(new ParameterShard(
        1,
        0.1,
        testWeights))

      //test layer
      val layerTestActor = TestActorRef(new Layer(
        1,
        1,
        (x: DenseVector[Double]) => x.map(el => sigmoid(el)),
        (x: DenseVector[Double]) => x.map(el => sigmoid(el) * (1 - sigmoid(el))),
        None,
        None,
        parameterShardTestActor
      ))

      //ask layer actor to update its weight parameter from its associated parameter shard
      layerTestActor ! FetchParameters

      layerTestActor.underlyingActor.latestWeights must equal(testWeights)
    }
  }


}
