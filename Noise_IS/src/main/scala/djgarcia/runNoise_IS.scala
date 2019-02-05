package main.scala.djgarcia

import java.io.PrintWriter

import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.RandomNoise
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object runNoise_IS extends Serializable {

  def main(arg: Array[String]) {

    //Basic setup
    val jobName = "MLlib Noise & IS"

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    val sc = new SparkContext(conf)

    var model: DecisionTreeModel = null
    var labelAndPreds: RDD[(Double, Double)] = sc.emptyRDD[(Double, Double)]
    var testAcc = 0.0

    //Load Train & Test

    val pathTrain = "file:///home/spark/datasets/susy-10k-tra.data"
    val rawDataTrain = sc.textFile(pathTrain)

    val pathTest = "file:///home/spark/datasets/susy-10k-tst.data"
    val rawDataTest = sc.textFile(pathTest)

    val train = rawDataTrain.map { line =>
      val array = line.split(",")
      var arrayDouble = array.map(f => f.toDouble)
      val featureVector = Vectors.dense(arrayDouble.init)
      val label = arrayDouble.last
      LabeledPoint(label, featureVector)
    }.persist

    train.count
    train.first

    val test = rawDataTest.map { line =>
      val array = line.split(",")
      var arrayDouble = array.map(f => f.toDouble)
      val featureVector = Vectors.dense(arrayDouble.init)
      val label = arrayDouble.last
      LabeledPoint(label, featureVector)
    }.persist

    test.count
    test.first


    //-----Noise-----//


    //Noisy Copy @ 20%

    val noise = 20 //(in %)

    val noisyModel = new RandomNoise(train, noise)

    val noisyData = noisyModel.runNoise()

    noisyData.persist()

    noisyData.count()


    //Decision Tree Clean Dataset

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 20 //Increased Depth
    val maxBins = 32

    model = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy Clean Dataset = $testAcc") //0.7058


    //Decision Tree Noisy Dataset

    model = DecisionTree.trainClassifier(noisyData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy Noisy Dataset = $testAcc") //0.6291


    //HME-BD Noise Filter

    val nTrees = 100
    val maxDepthRF = 10
    val partitions = 4

    val hme_bd_model_noisy = new HME_BD(noisyData, nTrees, partitions, maxDepthRF, 48151623)

    val hme_bd_noisy = hme_bd_model_noisy.runFilter()

    hme_bd_noisy.persist()

    hme_bd_noisy.count()


    //Decision Tree Filtered Dataset

    model = DecisionTree.trainClassifier(hme_bd_noisy, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy Filtered Dataset= $testAcc")


    //HME-BD Noise Filter with Clean Data

    val hme_bd_model_clean = new HME_BD(train, nTrees, partitions, maxDepthRF, 48151623)

    val hme_bd_clean = hme_bd_model_clean.runFilter()

    hme_bd_clean.persist()

    hme_bd_clean.count() //7814


    //Decision Tree Filtered Dataset

    model = DecisionTree.trainClassifier(hme_bd_clean, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy Filtered Dataset= $testAcc") //0.7947


    //NCNEdit Noise Filter

    val k = 3 //number of neighbors

    val ncnedit_bd_model = new NCNEdit_BD(noisyData, k)

    val ncnedit_bd = ncnedit_bd_model.runFilter()

    ncnedit_bd.persist()

    ncnedit_bd.count()


    //Decision Tree Filtered Dataset

    model = DecisionTree.trainClassifier(ncnedit_bd, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy NCNEdit= $testAcc")


    //RNG Noise Filter

    val order = true // Order of the graph (true = first, false = second)
    val selType = true // Selection type (true = edition, false = condensation)

    val rng_bd_model = new RNG_BD(noisyData, order, selType)

    val rng_bd = rng_bd_model.runFilter()

    rng_bd.persist()

    rng_bd.count()


    //Decision Tree Filtered Dataset


    model = DecisionTree.trainClassifier(rng_bd, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy RNG= $testAcc")


    //-----Instance Selection-----//


    //FCNN

    val fcnn_mr_model = new FCNN_MR(train, k)

    val fcnn_mr = fcnn_mr_model.runPR()

    fcnn_mr.persist()

    fcnn_mr.count()


    //Decision Tree FCNN

    model = DecisionTree.trainClassifier(fcnn_mr, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy FCNN = $testAcc")


    //SSMA-SFLSDE

    val ssmasflsde_mr_model = new SSMASFLSDE_MR(train)

    val ssmasflsde_mr = ssmasflsde_mr_model.runPR()

    ssmasflsde_mr.persist()

    ssmasflsde_mr.count()


    //Decision Tree SSMA-SFLSDE

    model = DecisionTree.trainClassifier(ssmasflsde_mr, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy SSMA-SFLSDE = $testAcc")


    //RMHC

    val p = 0.1 // Percentage of instances (max 1.0)
    val it = 100 // Number of iterations

    val rmhc_mr_model = new RMHC_MR(train, p, it, k, 48151623)

    val rmhc_mr = rmhc_mr_model.runPR()

    rmhc_mr.persist()

    rmhc_mr.count()


    // Decision Tree RMHC

    model = DecisionTree.trainClassifier(rmhc_mr, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy RMHC = $testAcc")

    //Write Results
    /*val writer = new PrintWriter("/home/user/results.txt")
    writer.write(
      "Test Acc: " + testAcc + "\n" +
        "Instances:  " + rmhc_mr.count() + "\n"
    )
    writer.close()*/
  }
}
