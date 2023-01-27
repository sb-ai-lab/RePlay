import os
import tempfile
from typing import Optional, Callable

from pyarrow import fs

from replay.utils import FileSystem


def save_index_to_destination_fs(
    index,
    sparse: bool,
    save_index: Callable[[str], None],
    filesystem: FileSystem,
    destination_path: str,
    hdfs_uri: Optional[str] = None,
):

    if filesystem == FileSystem.HDFS:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "index")
            destination_filesystem = fs.HadoopFileSystem.from_uri(hdfs_uri)

            save_index(temp_file_path)

            # here we copy index files from local disk to hdfs
            # if index is sparse then we need to copy two files: "index" and "index.dat"
            index_file_paths = [temp_file_path]
            destination_paths = [destination_path]
            if sparse:
                index_file_paths.append(temp_file_path + ".dat")
                destination_paths.append(destination_path + ".dat")
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
        save_index(destination_path)
