package cnn

import scala.io.Source
import com.intel.analytics.bigdl.{Module, nn, optim, utils}
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.slf4j.{Logger, LoggerFactory}

sealed trait Datum
final case class typeDF(df: sql.DataFrame) extends Datum
final case class typeArray(ar: Array[Array[String]]) extends Datum
final case class typeRDD(d: RDD[Array[String]]) extends Datum

class myCNN (val sparkSession: sql.SparkSession) {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  utils.LoggerFilter.redirectSparkInfoLogs()
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  def makeLeNet5(): nn.Sequential[Float] = nn.Sequential()
    .add(nn.Reshape(Array(1, 28, 28)))
    .add(nn.SpatialConvolution(1, 6, 5, 5))
    .add(nn.Tanh())
    .add(nn.SpatialMaxPooling(2, 2, 2, 2))
    .add(nn.SpatialConvolution(6, 12, 5, 5))
    .add(nn.Tanh())
    .add(nn.SpatialMaxPooling(2, 2, 2, 2))
    .add(nn.Reshape(Array(12 * 4 * 4)))
    .add(nn.Linear(12 * 4 * 4, 100))
    .add(nn.Tanh())
    .add(nn.Linear(100, 10))
    .add(nn.LogSoftMax())

  def normalizeData(tensor: Tensor[Float]): Tensor[Float] = tensor.div(255.0f)

  def setupOptimizer(
    model: nn.Sequential[Float],
    data: RDD[Sample[Float]],
    batchSize: Int,
    trainSplit: Double,
    maxEpochs: Int,
    learningRate: Double,
    momentum: Double,
    oneHotEncode: Boolean
  ): optim.Optimizer[Float, MiniBatch[Float]] = {
    val Array(train, validate) = data.randomSplit(weights = Array(trainSplit, 1 - trainSplit))
    val optimizer = optim.Optimizer(
      model = model,
      sampleRDD = train,
      batchSize = batchSize,
      criterion = if (oneHotEncode) nn.CategoricalCrossEntropy[Float]() else nn.CrossEntropyCriterion[Float]()
    )
    optimizer
      .setValidation(
        trigger = optim.Trigger.everyEpoch,
        // A big problem here is that no matter if you use CategoricalCrossEntropy or CrossEntropyCriterion, or any other
        // criterion, the validation requires labels and not one-hot tensors.
        sampleRDD = validate,
        batchSize = batchSize,
        vMethods = Array(new optim.Top1Accuracy[Float], new optim.Top5Accuracy[Float], new optim.Loss[Float])
      )
      .setOptimMethod(new optim.SGD[Float](learningRate = learningRate, momentum = momentum))
      .setEndWhen(optim.Trigger.maxEpoch(maxEpochs))
    optimizer
  }

  def runTestHarness(
    model: nn.Sequential[Float],
    dataFolder: String,
    trainName: String,
    testName: String,
    trainSplit: Double,
    maxEpochs: Int,
    batchSize: Int,
    learningRate: Double,
    momentum: Double,
    dataLoader: String,
    oneHotEncode: Boolean
  ): (Module[Float], Array[(optim.ValidationResult, optim.ValidationMethod[Float])]) = {
    def _readFile(path: String): Datum = dataLoader match {
      case "df" => typeDF(readFileDataFrame(path))
      case "array" => typeArray(readFileArray(path))
      case "rdd" => typeRDD(readFileRDD(path))
      case _ => throw new MatchError("dataLoader can only be 'array', 'df' or 'rdd' ")
    }

    log.info("Running test harness...")
    log.info("Reading data.")

    val trainData = _readFile(dataFolder + trainName)
    val testData = _readFile(dataFolder + testName)

    log.info("Processing data.")
    val train = processData(trainData, 28, 28, sc = sparkSession, oneHotEncode = oneHotEncode)
    val validate = processData(testData, 28, 28, sc = sparkSession, oneHotEncode = oneHotEncode)

    log.info("Setting up optimizer.")
    val optimizer = setupOptimizer(
      model,
      train,
      batchSize = batchSize,
      trainSplit = trainSplit,
      maxEpochs = maxEpochs,
      learningRate = learningRate,
      momentum = momentum,
      oneHotEncode = oneHotEncode
    )
    log.info("Running optimization...")
    val trainedModel = optimizer.optimize()

    log.info("Running evaluation.")
    val evaluationResult = trainedModel.evaluate(
      dataset = validate,
      vMethods = Array(
        new optim.Top1Accuracy[Float],
        new optim.Top5Accuracy[Float],
        new optim.Loss[Float]
      )
    )
    log.info("Test harness done!")
    (trainedModel, evaluationResult)
  }

  def readFileArray(path: String): Array[Array[String]] = {
    val src = Source.fromFile(path)
    val table = src.getLines()
                   .map(_.split(","))
                   .toArray
    src.close()
    table
  }

  def readFileRDD(path: String): RDD[Array[String]] = sparkSession.sparkContext
                                                                  .textFile(path)
                                                                  .map(_.split(","))

  def readFileDataFrame(
   path: String
 ): sql.DataFrame = sparkSession.read
    .format("csv")
    .options(Map("header" -> "true", "inferSchema" -> "true"))
    .load(path)


  def processRow(
    row: Array[String],
    dim1: Int,
    dim2: Int,
    oneHotEncode: Boolean = true
  ): Sample[Float] = {
    Sample(
      featureTensor = normalizeData(Tensor(data=row.tail.map(_.toFloat), shape=Array(dim1, dim2))),
      labelTensor = if (oneHotEncode) Tensor(10).zero.setValue(d1 = row(0).toInt + 1, value = 1f)
                    else Tensor(T(row.head.toInt + 1))
    )
  }

  def processData(
   data: Datum,
   dim1: Int,
   dim2: Int,
   sc: sql.SparkSession,
   oneHotEncode: Boolean = true
 ): RDD[Sample[Float]] = {
    sc.sparkContext.parallelize(data match {
      case typeDF(df) => df.collect.map(row =>
        processRow(row.toSeq.toArray.map(_.toString), dim1, dim2, oneHotEncode = oneHotEncode)
      )
      case typeArray(table) => table.tail.map(row =>
        processRow(row, dim1, dim2, oneHotEncode = oneHotEncode)
      )
      case typeRDD(rdd) => rdd.collect.drop(1).map(row =>
        processRow(row, dim1, dim2, oneHotEncode = oneHotEncode)
      )
      case _ => throw new MatchError("Data should be either in a DataFrame or Array format.")
    })
  }
}

object myCNN {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  utils.LoggerFilter.redirectSparkInfoLogs()
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  def initEngineAndSession(appName: String, master: String): sql.SparkSession = {
    log.info("Initializing Engine and Session")
    val conf = utils.Engine.createSparkConf()
      .setAppName(appName)
      .setMaster(master)
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    utils.Engine.init
    val session = sql.SparkSession.builder.config(sc.getConf).getOrCreate()
    log.info("Engine and Session ready.")
    session
  }

  def main(args: Array[String]): Unit = {
    println("start")
    //init spark context
    val cores = 12
    val sparkSession = initEngineAndSession(appName = "MNIST CNN", master = s"local[$cores]")
    val batchSize = cores * 3

    // https://pjreddie.com/projects/mnist-in-csv/
    val dataFolder = args(0)

    val trainName = "mnist_train.csv"
    val testName = "mnist_test.csv"

    val learningRate = 0.01
    val momentum = 0.9
    val trainSplit = 0.8
    val maxEpochs = 10

    val cnn = new myCNN(sparkSession = sparkSession)


    cnn.runTestHarness(
      model = cnn.defineModel(),
      dataFolder = dataFolder,
      trainName = trainName,
      testName = testName,
      batchSize = batchSize,
      trainSplit = trainSplit,
      maxEpochs = maxEpochs,
      learningRate = learningRate,
      momentum = momentum,
      dataLoader = "df",
      oneHotEncode = false
    )

    println("Done")
  }
}