Akka DistBelief
===============

DistBelief is a framework for training deep neural networks with a cluster of machines rather than GPUs.  The main
algorithm used is _Downpour SGD_ in which multiple replicas of the neural network model are trained in parallel.  Furthermore,
each replica is partitioned across machines by layer adding another level of parallelism.  The model parameters are
stored in a centralized server which is also partitioned across machines.  The model replica layers asynchronously
read and update their corresponding shard in the parameter server.

![downpour sgd](/images/downpour_sgd.png)

At its core, _Downpour SGD_ relies on asynchronous message passing which makes is a perfect fit for the Akka actor
 framework.  In this repo all partitions (i.e. data shards, parameter shards, model replica layers) are represented
 as Akka actors that communicate asynchronously via message passing.
 
 
 __Example__
 
An example using DistBelief to learn the non-linear XOR function using the sigmoid activation function.
 
| A             | B             | Output|
| ------------- |:-------------:| -----:|
| 0             | 0             |    0  |
| 0             | 1             |    1  |
| 1             | 0             |    1  |
| 1             | 1             |    0  |

```scala
class XOR extends Actor with ActorLogging{

  val random = new Random

  val possibleExamples = Seq(
    Example(DenseVector(1.0, 0.0, 0.0), DenseVector(0.0)),
    Example(DenseVector(1.0, 0.0, 1.0), DenseVector(1.0)),
    Example(DenseVector(1.0, 1.0, 0.0), DenseVector(1.0)),
    Example(DenseVector(1.0, 1.0, 1.0), DenseVector(0.0))
  )

  //generate 50000 training examples
  val trainingSet = (1 to 50000).foldLeft(Seq[Example]()) { (a, c) =>
    a :+ possibleExamples(random.nextInt(possibleExamples.size))
  }

  //create 25 model replicas each training 2000 data points in parallel
  val DistBeliefMaster = context.actorOf(Props(new Master(
    dataSet = trainingSet,
    dataPerReplica = 2000,
    layerDimensions = Seq(2, 2, 1),
    activation = (x: DenseVector[Double]) => x.map(el => sigmoid(el)),
    activationDerivative = (x: DenseVector[Double]) => x.map(el => sigmoid(el) * (1 - sigmoid(el))),
    learningRate = 0.5)))

  DistBeliefMaster ! Start

  def receive = {
    case JobDone => log.info("Finished Computing XOR Example!!!!!")
  }
}
```
 
 __References__
 
 * _Large Scale Distributed Deep Networks_.  Jeffrey Dean, Greg S. Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Quoc V. Le, Mark Z. Mao, Marc’Aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang and Andrew Y. Ng.  NIPS 2012.
