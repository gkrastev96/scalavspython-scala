name := "scala-cnn-2.12.0"

version := "0.1"

scalaVersion := "2.12.0"

// import spark
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.1.2"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.1.2"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.1.2"

// Importing BigDL library
libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-SPARK_3.0" % "0.13.0"