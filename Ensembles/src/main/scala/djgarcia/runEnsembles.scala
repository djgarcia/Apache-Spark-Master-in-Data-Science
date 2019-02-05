package main.scala.djgarcia

import java.io.PrintWriter

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, PCARD, RandomForest}
import org.apache.spark.{SparkConf, SparkContext}

object runEnsembles {

  def main(arg: Array[String]) {

    //Basic setup
    val jobName = "MLlib Ensembles"

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    val sc = new SparkContext(conf)

    //Load train and test

    val pathTrain = "file:///home/spark/datasets/susy-10k-tra.data"
    val rawDataTrain = sc.textFile(pathTrain)

    val pathTest = "file:///home/spark/datasets/susy-10k-tst.data"
    val rawDataTest = sc.textFile(pathTest)

    val train = rawDataTrain.map { line =>
      val array = line.split(",")
      val arrayDouble = array.map(f => f.toDouble)
      val featureVector = Vectors.dense(arrayDouble.init)
      val label = arrayDouble.last
      LabeledPoint(label, featureVector)
    }.persist

    train.count
    train.first

    val test = rawDataTest.map { line =>
      val array = line.split(",")
      val arrayDouble = array.map(f => f.toDouble)
      val featureVector = Vectors.dense(arrayDouble.init)
      val label = arrayDouble.last
      LabeledPoint(label, featureVector)
    }.persist

    test.count
    test.first


    //Class balance

    val classInfo = train.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()


    //Decision tree

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    var numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    var impurity = "gini"
    var maxDepth = 5
    var maxBins = 32

    val modelDT = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPredsDT = test.map { point =>
      val prediction = modelDT.predict(point.features)
      (point.label, prediction)
    }
    val testAccDT = 1 - labelAndPredsDT.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy DT= $testAccDT")


    //Random Forest

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    numClasses = 2
    categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 100
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    impurity = "gini"
    maxDepth = 4
    maxBins = 32

    val modelRF = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPredsRF = test.map { point =>
      val prediction = modelRF.predict(point.features)
      (point.label, prediction)
    }
    val testAccRF = 1 - labelAndPredsRF.filter(r => r._1 != r._2).count.toDouble / test.count()
    println(s"Test Accuracy RF= $testAccRF")


    //PCARD

    val cuts = 5
    val trees = 10

    val pcardTrain = PCARD.train(train, trees, cuts)

    val pcard = pcardTrain.predict(test)


    def avgAcc(labels: Array[Double], predictions: Array[Double]): (Double, Double) = {
      var cont = 0
      for (i <- labels.indices) {
        if (labels(i) == predictions(i)) {
          cont += 1
        }
      }
      (cont / labels.length.toFloat, 1 - cont / labels.length.toFloat)
    }

    print("PCARD Accuracy: " + avgAcc(test.map(_.label).collect(), pcard)._1)

    val rdd_labels = sc.parallelize(pcard).zipWithIndex.map { case (v, k) => (k, v) }

    val labelAndPreds = test.zipWithIndex.map { case (v, k) => (k, v.label) }.join(rdd_labels).map(_._2)

    //Metrics

    import org.apache.spark.mllib.evaluation.MulticlassMetrics

    val metrics = new MulticlassMetrics(labelAndPreds)
    val precision = metrics.precision
    val cm = metrics.confusionMatrix


    //Write Results
    /*val writer = new PrintWriter("/home/user/results.txt")
    writer.write(
      "Precision: " + precision + "\n" +
        "Confusion Matrix " + cm + "\n"
    )
    writer.close()*/
  }
}
