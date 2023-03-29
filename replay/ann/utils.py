import os
import tempfile
from typing import Optional, Callable

from pyarrow import fs

from replay.utils import FileSystem, FileInfo


def save_index_to_destination_fs(
    sparse: bool,
    save_index: Callable[[str], None],
    target: FileInfo,
):

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
