# Get Started

## Data Format

RePlay uses PySpark for internal data representation. 
To convert Pandas dataframe into a spark one you can use `replay.utils.spark_utils.convert2spark` function.

By default you don't have to think about Spark session at all because it will be created automatically.
If you want to use custom session, refer to [this](spark.rst) page.

There are also requirements regarding column names.

| Entity             | Name      |
| ------------------ | --------- |
| User identificator | user_idx  |
| Item identificator | item_idx  |
| Date info          | timestamp |
| Rating/weight      | relevance |

### ID requirements

user_idx and item_idx should be numerical indexes starting at zero without gaps. 
This is important for models that use sparse matrices and estimate their dimensions on biggest seen index.

You should convert your data with [Indexer](modules/preprocessing.rst) class. 
It will store label encoders for you to convert raw id to idx and back.

### Timestamp requirements

timestamp can be integer, but it is preferable if it is a datetime timestamp.
