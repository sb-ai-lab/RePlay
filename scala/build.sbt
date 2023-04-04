import sbt.Keys.resolvers

name := "replay"

version := "0.1"

scalaVersion := "2.12.15"

// idePackagePrefix := Some("org.apache.spark.ml.feature.lightautoml")

resolvers ++= Seq(
  ("Confluent" at "http://packages.confluent.io/maven")
        .withAllowInsecureProtocol(true)
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.1.3",
  "org.apache.spark" %% "spark-sql" % "3.1.3",
  "org.apache.spark" %% "spark-mllib" % "3.1.3",
)