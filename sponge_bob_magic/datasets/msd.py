import os
import pandas as pd
from os.path import join

from sponge_bob_magic.datasets.data_loader import download_msd
from sponge_bob_magic.datasets.generic_dataset import Dataset


class MillionSongDataset(Dataset):
    """
    Враппер для MSD, обеспечивает загрузку и парсинг.

    Данный датасет, так же называемый Echo Nest Taste Profile Subset --
    часть MSD, содержащая информацию о прослушиваниях треков пользователями.

    - 1,019,318 уникальных пользователей
    - 384,546 уникальных песен
    - 48,373,586 троек user - song - play count

    Пример загрузки:
    >>> from sponge_bob_magic.datasets import MillionSongDataset
    >>> msd = MillionSongDataset()
    >>> msd.info()
    train
                                        user_id             item_id  relevance
    0  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SOAKIMP12A8C130995          1
    1  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SOAPDEY12A81C210A9          1
    2  b80344d063b5ccb3212f76538f3d9e43d87dca9e  SOBBMDR12A8C13253B          2
    val
                                        user_id             item_id  relevance
    0  0007140a3796e901f3190f12e9de6d7548d4ac4a  SONVMBN12AC9075271          1
    1  0007140a3796e901f3190f12e9de6d7548d4ac4a  SOVIGZG12A6D4FB188          1
    2  0007140a3796e901f3190f12e9de6d7548d4ac4a  SOZGXYF12AB0185579          2
    test
                                        user_id             item_id  relevance
    0  00007a02388c208ea7176479f6ae06f8224355b3  SOAITVD12A6D4F824B          3
    1  00007a02388c208ea7176479f6ae06f8224355b3  SONZGLW12A6D4FBBC1          1
    2  00007a02388c208ea7176479f6ae06f8224355b3  SOXNWYP12A6D4FBDC4          1

    Подробнее: http://millionsongdataset.com/tasteprofile/
    """
    def __init__(self, merge_kaggle_splits: bool = True, drop_mismatches: bool = True):
        """Загрузить все в память.
        ! Датасет большой и на маке в первый раз грузится минут пять,
        ! суммарно занимая в оперативке около 1.2GB
        ! При параметрах по умолчанию, результат записывается на диск
        ! и последующие загрузки занимают полминуты.

        :param merge_kaggle_splits: bool
            В MSD Challenge на кэггл была паблик и прайват часть,
            эти файлы разделены, но по умолчанию они сливаются вместе.
        :param drop_mismatches: bool
            Существует ошибка в соотношении песен и треков в MSD.
            По умолчанию эти песни выкидываются из датасета.
            Подробнее: http://millionsongdataset.com/blog/12-2-12-fixing-matching-errors/
        """
        super().__init__()
        folder = join(self.data_folder, "msd")
        if not os.path.exists(folder):
            download_msd(self.data_folder)

        try_cache = merge_kaggle_splits and drop_mismatches
        processed = join(folder, "clean")
        if try_cache and os.path.exists(processed):
            self.train = pd.read_csv(join(processed, "train.csv"))
            self.val = pd.read_csv(join(processed, "val.csv"))
            self.test = pd.read_csv(join(processed, "test.csv"))
        else:
            self.train = self._read_triplets(join(folder, "train_triplets.txt"))
            val_vis = self._read_triplets(join(folder, "evaluation",
                                               "year1_valid_triplets_visible.txt"))
            val_hid = self._read_triplets(join(folder, "evaluation",
                                               "year1_valid_triplets_hidden.txt"))
            test_vis = self._read_triplets(join(folder, "evaluation",
                                                "year1_test_triplets_visible.txt"))
            test_hid = self._read_triplets(join(folder, "evaluation",
                                                "year1_test_triplets_hidden.txt"))
            if drop_mismatches:
                mismatches = self._read_mismatches(folder)
                mismatches = set(mismatches.item_id)
                self.train = self._drop_mismatches(self.train, mismatches)
                val_vis = self._drop_mismatches(val_vis, mismatches)
                val_hid = self._drop_mismatches(val_hid, mismatches)
                test_vis = self._drop_mismatches(test_vis, mismatches)
                test_hid = self._drop_mismatches(test_hid, mismatches)

            if merge_kaggle_splits:
                self.val = pd.concat([val_vis, val_hid], ignore_index=True)
                self.test = pd.concat([test_vis, test_hid], ignore_index=True)
            else:
                self.val_visible = val_vis
                self.val_hidden = val_hid
                self.test_visible = test_vis
                self.test_hidden = test_hid

            if try_cache and not os.path.exists(processed):
                os.mkdir(processed)
                self.train.to_csv(join(processed, "train.csv"), index=False)
                self.val.to_csv(join(processed, "val.csv"), index=False)
                self.test.to_csv(join(processed, "test.csv"), index=False)

    @staticmethod
    def _read_triplets(path):
        return pd.read_csv(path, names=["user_id", "item_id", "relevance"],
                           sep="\t", dtype={"user_id": "category",
                                            "item_id": "category"}).dropna()

    @staticmethod
    def _read_mismatches(path):
        name = "sid_mismatches.txt"
        file = join(path, name)
        mismatches = []
        with open(file) as f:
            for line in f.readlines():
                song, track = line[line.find("<") + 1: line.find(">")].split(" ")
                mismatches.append([song, track])
        return pd.DataFrame(mismatches, columns=["item_id", "track_id"])

    @staticmethod
    def _drop_mismatches(df, mismatches):
        return df[~df.item_id.isin(mismatches)]
