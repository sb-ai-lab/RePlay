import os
import tempfile
from typing import Callable

import hnswlib
import nmslib
from pyarrow import fs

from replay.ann.entities.hnswlib_param import HnswlibParam
from replay.ann.entities.nmslib_hnsw_param import NmslibHnswParam
from replay.utils import FileSystem, FileInfo


def save_index_to_destination_fs(
    sparse: bool,
    save_index: Callable[[str], None],
    target: FileInfo,
):
    """
    Function saves `index` to destination filesystem (local disk or hdfs).
    If destination filesystem is HDFS then `index` dumps to temporary path in local disk,
    and then copies to HDFS.
    :param sparse: flag, index is sparse or not.
    :param save_index: lambda expression that dumps index to local disk.
    :param target: destination filesystem properties.
    :return:
    """

    if target.filesystem == FileSystem.HDFS:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "index")
            destination_filesystem = fs.HadoopFileSystem.from_uri(
                target.hdfs_uri
            )

            save_index(temp_file_path)

            # here we copy index files from local disk to hdfs
            # if index is sparse then we need to copy two files: "index" and "index.dat"
            index_file_paths = [temp_file_path]
            destination_paths = [target.path]
            if sparse:
                index_file_paths.append(temp_file_path + ".dat")
                destination_paths.append(target.path + ".dat")
            for index_file_path, destination_path in zip(
                index_file_paths, destination_paths
            ):
                fs.copy_files(
                    "file://" + index_file_path,
                    destination_path,
                    destination_filesystem=destination_filesystem,
                )
                # param use_threads=True (?)
    else:
        save_index(target.path)


def load_index_from_source_fs(
    sparse: bool,
    load_index: Callable[[str], None],
    source: FileInfo,
):
    """
    Function loads `index` from source filesystem (local disk or hdfs).
    This function loads `index` that was saved via `save_index_to_destination_fs`.
    :param sparse: flag, index is sparse or not.
    :param load_index: lambda expression that loads index from local disk.
    :param source: source filesystem properties.
    :return:
    """
    if source.filesystem == FileSystem.HDFS:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "index")
            source_filesystem = fs.HadoopFileSystem.from_uri(source.hdfs_uri)

            # here we copy index files from hdfs to local disk
            # if index is sparse then we need to copy two files: "index" and "index.dat"
            index_file_paths = [source.path]
            destination_paths = [temp_file_path]
            if sparse:
                index_file_paths.append(source.path + ".dat")
                destination_paths.append(temp_file_path + ".dat")
            for index_file_path, destination_path in zip(
                index_file_paths, destination_paths
            ):
                fs.copy_files(
                    index_file_path,
                    "file://" + destination_path,
                    source_filesystem=source_filesystem,
                )
            load_index(temp_file_path)
    elif source.filesystem == FileSystem.LOCAL:
        load_index(source.path)


def create_hnswlib_index_instance(params: HnswlibParam, init: bool = False):
    """
    Creates and returns hnswlib index

    :param params: `HnswlibParam`
    :param init: If `True` it will call the `init_index` method on the index.
        Used when we want to create a new index.
        If `False` then the index will be used to load index data from a file.
    :return: `hnswlib` index instance
    """
    index = hnswlib.Index(  # pylint: disable=c-extension-no-member
        space=params.space, dim=params.dim
    )

    if init:
        # Initializing index - the maximum number of elements should be known beforehand
        index.init_index(
            max_elements=params.max_elements,
            ef_construction=params.efC,
            M=params.M,
        )

    return index


def create_nmslib_index_instance(params: NmslibHnswParam):
    """
    Creates and returns nmslib index

    :param params: `NmslibHnswParam`
    :return: `nmslib` index
    """
    index = nmslib.init(  # pylint: disable=c-extension-no-member
        method=params.method,
        space=params.space,
        data_type=nmslib.DataType.SPARSE_VECTOR,  # pylint: disable=c-extension-no-member
    )

    return index
