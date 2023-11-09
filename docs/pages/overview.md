# Get Started

## Data Format

RePlay uses PySpark for internal data representation. 
To convert Pandas dataframe into a spark one you can use `replay.utils.spark_utils.convert2spark` function.

By default you don't have to think about Spark session at all because it will be created automatically.
If you want to use custom session, refer to [this](spark.rst) page.


### Timestamp requirements

timestamp can be integer, but it is preferable if it is a datetime timestamp.
