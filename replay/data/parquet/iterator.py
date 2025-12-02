from collections.abc import Iterator
from typing import Any, Callable, Optional

import pyarrow as pa
import pyarrow.dataset as da
import torch

from replay.constants.device import DEFAULT_DEVICE
from replay.data.parquet.impl.masking import DEFAULT_MAKE_MASK_NAME

from .impl.array_1d_column import to_array_1d_columns
from .impl.array_2d_column import to_array_2d_columns
from .impl.flat_column import to_flat_columns
from .impl.named_columns import NamedColumns
from .metadata import Metadata


class BatchesIterator:
    """
    Итератор для побатчевого извлечения данных из parquet-датасета с преобразованием в структурированные колонки.

    Аргументы:
        metadata (Metadata): Метаданные, описывающие структуру и типы данных.
        dataset (da.Dataset): Pyarrow-датасет, поддерживающий метод to_batches.
        batch_size (int): Размер батча при обработке одной партиции parquet-датасета.
            Размер получаемого батча не всегда будет равен batch_size.
            По причине того, что в фрагменте parquet-файла может содержаться количество строк некратное batch_size.
            Например, если фрагмент содержит 1000 строк и batch_size равен 64,
            то вы получите 15 батчей размера 64 и последний батч будет размером 40.
        make_mask_name (Callable[[str], str], optional): Функция для генерации имени маски.
            По умолчанию `DEFAULT_MAKE_MASK_NAME`.
        device (torch.device, optional): Устройство для хранения тензоров (CPU/GPU). По умолчанию `DEFAULT_DEVICE`.
        pyarrow_kwargs (Dict[str, Any], optional): Дополнительные аргументы для метода `to_batches`.
            Для большего понимания смотрите документацию метода `to_batches` в PyArrow Dataset.

    Атрибуты:
        dataset (da.Dataset): Входной датасет.
        metadata (Metadata): Метаданные.
        batch_size (int): Размер батча.
        make_mask_name (Callable[[str], str]): Функция формирования имени маски.
        device (torch.device): Устройство, на котором будут формироваться данные.
        pyarrow_kwargs (Dict[str, Any]): Параметры для PyArrow.
    """

    def __init__(
        self,
        metadata: Metadata,
        dataset: da.Dataset,
        batch_size: int,
        make_mask_name: Callable[[str], str] = DEFAULT_MAKE_MASK_NAME,
        device: torch.device = DEFAULT_DEVICE,
        pyarrow_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if pyarrow_kwargs is None:
            pyarrow_kwargs = {}
        self.dataset: da.Dataset = dataset
        self.metadata: Metadata = metadata
        self.batch_size: int = batch_size
        self.make_mask_name: Callable[[str], str] = make_mask_name
        self.device: torch.device = device
        self.pyarrow_kwargs: dict[str, Any] = pyarrow_kwargs

    def __iter__(self) -> Iterator[NamedColumns]:
        batch: pa.RecordBatch
        for batch in self.dataset.to_batches(
            batch_size=self.batch_size,
            columns=list(self.metadata.keys()),
            **self.pyarrow_kwargs,
        ):
            yield NamedColumns(
                columns={
                    **to_flat_columns(batch, self.metadata, self.device),
                    **to_array_1d_columns(batch, self.metadata, self.device),
                    **to_array_2d_columns(batch, self.metadata, self.device),
                },
                make_mask_name=self.make_mask_name,
            )
