#!/bin/bash
#Local
/opt/spark-2.2.0-bin-hadoop2.7/bin/spark-submit --master local[*] --class main.scala.djgarcia.runEnsembles ./target/Ensembles-1.0-jar-with-dependencies.jar

#Cluster (.jar must be in cluster)
#/opt/spark-2.2.0/bin/spark-submit --master spark://hadoop-master:7077 --class main.scala.djgarcia.runEnsembles Ensembles-1.0-jar-with-dependencies.jar
