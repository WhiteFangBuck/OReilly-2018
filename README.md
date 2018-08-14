# Oreilly - 2018

This tutorial can either be run in spark-shell or in an IDE (IntelliJ or Scala IDE for Eclipse)

Below are the steps for the setup.

## Pre-requisites for Installation

Java/JDK 1.8+ has to be installed on the laptop before proceeding with the steps below.

## Running in spark-shell

### Download Spark 2.3.0

Download Spark 2.3.0 from here : http://spark.apache.org/downloads.html

Direct Download link : https://www.apache.org/dyn/closer.lua/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz

### Install Spark 2.2.0 on Mac/Linux

tar -zxvf spark-2.3.1-bin-hadoop2.7.tgz

export PATH=$PATH:/<path_to_downloaded_spark>/spark-2.3.1-bin-hadoop2.7/bin

### Running spark-shell on mac

- spark-shell

### Install Spark 2.3.1 on Windows

Unzip spark-2.3.1-bin-hadoop2.7.tgz

Add the spark bin directory to Path : ...\spark-2.3.1-bin-hadoop2.7\bin

### Set up winutils.exe on Windows (not needed on mac)

- download winutils.exe from https://github.com/steveloughran/winutils/tree/master/hadoop-2.6.0/bin
- move it to c:\hadoop\bin
- set HADOOP_HOME in your environment variables
    - HADOOP_HOME = C:\hadoop
- run from command prompt:
    - mkdir \tmp\hive
    - C:\hadoop\bin\winutils.exe chmod 777 \tmp\hive
- run spark-shell from command prompt with extra conf parameter
    - spark-shell --driver-memory 2G --executor-memory 3G --executor-cores 2 --conf spark.sql.warehouse.dir=file:///c:/tmp/spark-warehouse


### Pasting code in spark-shell

When pasting larger sections of the code in spark-shell, use the below:

scala> :paste

## Running in IDE

If you prefer to use IDE over spark-shell, below are the steps.

You can either use IntelliJ or Scala IDE for Eclipse.

### IntelliJ

- Install IntelliJ from https://www.jetbrains.com/idea/download/
- Add the scala language plugin
- Import the code as a maven project and let it build

### Scala IDE for Eclipse

- If using Eclipse, do use Scala IDE for Eclipse available at : http://scala-ide.org/download/sdk.html
- Import the code as a maven project and let it build

## Summary of Downloads needed

Have the following downloaded before the session

- JDK installed (> 1.8.x)
- Spark binaries
- https://github.com/WhiteFangBuck/OReilly-2018


## Git

Nice to have


hadoop fs -copyToLocal  /strata-nyc/transferlearning.tgz .


tar -zxvf transferlearning.tgz

pip install tensorflow  

pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_64.whl

pip install numpy scipy

pip install scikit-learn

pip install pillow

pip install h5py

pip install keras


