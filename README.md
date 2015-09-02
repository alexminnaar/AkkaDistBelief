#Akka DistBelief

DistBelief is a framework for training deep neural networks with a cluster of machines rather than GPUs.  The main
algorithm used is _Downpour SGD_ in which multiple replicas of the neural network model are trained in parallel.  Furthermore,
each replica is partitioned across machines by layer adding another level of parallelism.  The model parameters are
stored in a centralized server which is also partitioned across machines.  The model replica layers asynchronously
read and update their corresponding shard in the parameter server.

![downpour sgd](/downpour_sgd.png)

At its core, _Downpour SGD_ relies asynchronous message passing which makes is a perfect fit for the Akka actor
 framework.  In this repo all partitions (i.e. data shards, parameter shards, model replica layers) are represented
 as Akka actors that communicate asynchronously via message passing.
 
 #References
 
 * "Large Scale Distributed Deep Networks".  Jeffrey Dean and Greg S. Corrado and Rajat Monga and Kai Chen and Matthieu Devin and Quoc V. Le and Mark Z. Mao and Marc’Aurelio Ranzato and Andrew Senior and Paul Tucker and Ke Yang and Andrew Y. Ng.  NIPS 2012.