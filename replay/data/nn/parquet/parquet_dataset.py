import warnings
from collections.abc import Callable, Iterator
from typing import Optional, Union, cast

import pyarrow.dataset as ds
import pyarrow.fs as fs
import torch
from torch.utils.data import IterableDataset

from replay.constants.batches import GeneralBatch
from replay.constants.device import DEFAULT_DEVICE
from replay.constants.filesystem import DEFAULT_FILESYSTEM
from replay.data.nn.parquet.impl.masking import (
    DEFAULT_COLLATE_FN,
    DEFAULT_MAKE_MASK_NAME,
    DEFAULT_REPLICAS_INFO,
    GeneralCollateFn,
)
from replay.data.nn.parquet.info.replicas import ReplicasInfoProtocol
from replay.data.nn.parquet.utils.compute_length import compute_fixed_size_length

from .fixed_batch_dataset import FixedBatchSizeDataset
from .iterator import BatchesIterator
from .metadata import Metadata
from .partitioned_iterable_dataset import PartitionedIterableDataset


class ParquetDataset(IterableDataset):
    """
    Датасет для загрузки данных из одного или нескольких Parquet-файлов с поддержкой батчевания.

    Эта реализация позволяет читать данные с помощью PyArrow Dataset, преобразовывать их в структурированные колонки,
        разбивать на партиции, а затем на батчи, необходимые для обучения модели.
        Поддерживает распределённое обучение и воспроизводимое случайное перемешивание.
    Во время работы даталоадера читается партиция размером `partition_size`.
        Могут возникать ситуации, когда размер прочитанной партиции будет меньше,
        чем `partition_size` - это зависит от количества строк в фрагменте данных.
        Фрагмент - это один Parquet-файл в файловой системе.
    Прочитанная партиция будет обработана и результат выдан в виде батча размером `batch_size`.
    Обратите внимание, что размер батча в результате может быть меньше `batch_size`.
    Для получения наибольшей эффективности при чтении и обработке данных
        рекомендуется устанавливать `partition_size` в несколько раз больше, чем `batch_size`.

    Аргументы:
        source (Union[str, List[str]]): Путь или список путей к файлам/каталогам с данными в формате Parquet.
        metadata (Metadata): Метаданные, описывающие структуру данных.
            Структура каждого столбца определяется следующими значениями:
            `shape` - размерность прочитываемого столбца.
                Если столбец содержит только одно значение, то параметр указывать не нужно.
                Если столбец содержит одномерный массив, то параметр должен быть числом или массивом,
                    содержащим одно число.
                Если столбец содержит двумерный массив, то параметр должен быть массивом, содержащим 2 числа.
            `padding` - паддинг значение, которым будут заполняться массивы в случае если его длина меньше,
                чем указанная в параметре `shape`.
        partition_size (int): Размер партиции при чтении данных из Parquet-файлов.
        batch_size (int): Размер батча, который будет возвращаться при итерации.
        filesystem (Union[str, fs.FileSystem], optional): Файловая система для доступа к данным.
            По умолчанию DEFAULT_FILESYSTEM.
        make_mask_name (Callable[[str], str], optional): Функция для генерации имени маски.
            По умолчанию DEFAULT_MAKE_MASK_NAME.
        device (torch.device, optional): Устройство для хранения тензоров. По умолчанию DEFAULT_DEVICE.
        generator (Optional[torch.Generator], optional): Генератор случайных чисел для перемешивания батчей.
        replicas_info (ReplicasInfoProtocol, optional): Информация о репликах. По умолчанию DEFAULT_REPLICAS_INFO.
        collate_fn (GeneralCollateFn, optional): Функция для соединения батчей. По умолчанию DEFAULT_COLLATE_FN.
        kwargs (Dict[str, Any]): Дополнительные аргументы.
            Для передачи дополнительных аргументов в PyArrow Dataset
            необходимо передать аргументы в виде словаря с ключом `pyarrow_dataset_kwargs`.
            Для передачи дополнительных аргументов в метод `to_batches` в PyArrow Dataset
            необходимо передать аргументы в виде словаря с ключом `pyarrow_to_batches_kwargs`.
        pyarrow_dataset_kwargs (Dict[str, Any], optional): Дополнительные параметры для создания PyArrow Dataset.
        pyarrow_to_batches_kwargs (Dict[str, Any], optional): Дополнительные параметры для метода to_batches
            в PyArrow Dataset.


    Атрибуты:
        filesystem (fs.FileSystem): Файловая система для доступа к данным.
        metadata (Metadata): Метаданные, описывающие структуру данных.
        batch_size (int): Размер батча.
        partition_size (int): Размер партиции при чтении данных из Parquet-файлов.
        replicas_info (ReplicasInfoProtocol): Информация о репликах.
        iterator (BatchesIterator): Итератор для чтения партиций из PyArrow Dataset.
        raw_dataset (PartitionedIterableDataset): Датасет, который обрабатывает итератор и возвращает батчи.
        dataset (FixedBatchSizeDataset): Объект датасета, поддерживающий итерацию по батчам фиксированного размера.
        do_compute_length (bool): Флаг вычисления длины датасета. По умолчанию `True`.
            Это может быть медленно в некоторых случаях.
    """

    def __init__(
        self,
        source: Union[str, list[str]],
        metadata: Metadata,
        partition_size: int,
        batch_size: int,
        filesystem: Union[str, fs.FileSystem] = DEFAULT_FILESYSTEM,
        make_mask_name: Callable[[str], str] = DEFAULT_MAKE_MASK_NAME,
        device: torch.device = DEFAULT_DEVICE,
        generator: Optional[torch.Generator] = None,
        replicas_info: ReplicasInfoProtocol = DEFAULT_REPLICAS_INFO,
        collate_fn: GeneralCollateFn = DEFAULT_COLLATE_FN,
        **kwargs,
    ) -> None:
        if partition_size < batch_size:
            msg = (
                "Suboptimal parameters: partition size is smaller than batch size. "
                f"Got: {partition_size=}, {batch_size=}."
            )
            warnings.warn(msg, stacklevel=2)

        if (partition_size % batch_size) != 0:
            msg = (
                "Suboptimal parameters: partition size is not multiple of batch size. "
                f"Got: {partition_size=}, {batch_size=}."
            )
            warnings.warn(msg, stacklevel=2)

        if isinstance(filesystem, str):
            filesystem, _ = fs.FileSystem.from_uri(filesystem)
        assert isinstance(filesystem, fs.FileSystem)
        self.filesystem = cast(fs.FileSystem, filesystem)

        self.pyarrow_dataset = ds.dataset(
            source,
            filesystem=self.filesystem,
            format="parquet",
            **kwargs.get("pyarrow_dataset_kwargs", {}),
        )

        self.batch_size = batch_size
        self.partition_size = partition_size
        self.replicas_info = replicas_info
        self.metadata = metadata

        self.iterator = BatchesIterator(
            dataset=self.pyarrow_dataset,
            metadata=self.metadata,
            batch_size=partition_size,
            device=device,
            make_mask_name=make_mask_name,
            pyarrow_kwargs=kwargs.get("pyarrow_to_batches_kwargs", {}),
        )

        self.raw_dataset = PartitionedIterableDataset(
            batch_size=batch_size,
            iterable=self.iterator,
            generator=generator,
            replicas_info=replicas_info,
        )

        self.dataset = FixedBatchSizeDataset(
            dataset=self.raw_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        self.do_compute_length: bool = True
        self.cached_lengths: dict[int, int] = {}

    def compute_length(self) -> int:
        """
        Возвращает длину датасета в батчах фиксированного размера.
        """
        num_replicas: int = self.replicas_info.num_replicas
        if num_replicas not in self.cached_lengths:
            if len(self.cached_lengths) > 0:
                msg = "`num_replicas` changed. Unable to reuse cached length."
                warnings.warn(msg, stacklevel=2)
            curr_length = compute_fixed_size_length(
                iterable=self.iterator,
                num_replicas=num_replicas,
                batch_size=self.batch_size,
            )
            self.cached_lengths[num_replicas] = curr_length
        return self.cached_lengths[num_replicas]

    def __len__(self) -> int:
        if self.do_compute_length:
            return self.compute_length()
        msg = "This instance doesn't support `len()` method. You can enable it by setting `do_compute_length=True`."
        raise TypeError(msg)

    def __iter__(self) -> Iterator[GeneralBatch]:
        return iter(self.dataset)
