from dataclasses import dataclass
from enum import Enum

from pyspark.sql import SparkSession


class FileSystem(Enum):
    """File system types"""
    HDFS = 1
    LOCAL = 2


def get_default_fs() -> str:
    """Returns hadoop `fs.defaultFS` property value"""
    spark = SparkSession.getActiveSession()
    hadoop_conf = spark._jsc.hadoopConfiguration()
    default_fs = hadoop_conf.get("fs.defaultFS")
    return default_fs


@dataclass(frozen=True)
class FileInfo:
    """File meta-information: filesystem, path and hdfs_uri (optional)"""
    path: str
    filesystem: FileSystem
    hdfs_uri: str = None


def get_filesystem(path: str) -> FileInfo:
    """Analyzes path and hadoop config and return `FileInfo` instance with `filesystem`,
    `hdfs uri` (if filesystem is hdfs) and `cleaned path` (without prefix).

    For example:

    >>> path = 'hdfs://node21.bdcl:9000/tmp/file'
    >>> get_filesystem(path)
    FileInfo(path='/tmp/file', filesystem=<FileSystem.HDFS: 1>, hdfs_uri='hdfs://node21.bdcl:9000')

    >>> path = 'file:///tmp/file'
    >>> get_filesystem(path)
    FileInfo(path='/tmp/file', filesystem=<FileSystem.LOCAL: 2>, hdfs_uri=None)

    >>> spark = SparkSession.builder.master("local[1]").getOrCreate()
    >>> path = 'hdfs:///tmp/file'
    >>> get_filesystem(path)
    Traceback (most recent call last):
     ...
    ValueError: Can't get default hdfs uri for path = 'hdfs:///tmp/file'. \
Specify an explicit path, such as 'hdfs://host:port/dir/file', \
or set 'fs.defaultFS' in hadoop configuration.

    >>> path = '/tmp/file'
    >>> get_filesystem(path)
    FileInfo(path='/tmp/file', filesystem=<FileSystem.LOCAL: 2>, hdfs_uri=None)

    >>> spark = SparkSession.builder.master("local[1]").getOrCreate().newSession()
    >>> spark.sparkContext._jsc.hadoopConfiguration().set('fs.defaultFS', 'hdfs://node21.bdcl:9000')
    >>> path = '/tmp/file'
    >>> get_filesystem(path)
    FileInfo(path='/tmp/file', filesystem=<FileSystem.HDFS: 1>, hdfs_uri='hdfs://node21.bdcl:9000')

    >>> path = 'hdfs:///tmp/file'
    >>> get_filesystem(path)
    FileInfo(path='/tmp/file', filesystem=<FileSystem.HDFS: 1>, hdfs_uri='hdfs://node21.bdcl:9000')

    Return fs.defaultFS value because the current session may be used in another test
    >>> spark.sparkContext._jsc.hadoopConfiguration().set('fs.defaultFS', 'file:///')


    Args:
        path (str): path to file on hdfs or local disk

    Returns: FileInfo instance: `filesystem id`,
    `hdfs uri` (if filesystem is hdfs) and `cleaned path` (without prefix)
    """
    prefix_len = 7  # 'hdfs://' and 'file://' length
    if path.startswith("hdfs://"):
        if path.startswith("hdfs:///"):
            default_fs = get_default_fs()
            if default_fs.startswith("hdfs://"):
                return FileInfo(path[prefix_len:], FileSystem.HDFS, default_fs)
            else:
                raise ValueError(
                    f"Can't get default hdfs uri for path = '{path}'. "
                    "Specify an explicit path, such as 'hdfs://host:port/dir/file', "
                    "or set 'fs.defaultFS' in hadoop configuration."
                )
        else:
            hostname = path[prefix_len:].split("/", 1)[0]
            hdfs_uri = "hdfs://" + hostname
            return FileInfo(path[len(hdfs_uri):], FileSystem.HDFS, hdfs_uri)
    elif path.startswith("file://"):
        return FileInfo(path[prefix_len:], FileSystem.LOCAL)
    else:
        default_fs = get_default_fs()
        if default_fs.startswith("hdfs://"):
            return FileInfo(path, FileSystem.HDFS, default_fs)
        else:
            return FileInfo(path, FileSystem.LOCAL)
