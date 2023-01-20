import os

from pyspark.ml.util import DefaultParamsWriter
from pyspark.sql import DataFrame


class DataframeAwareDefaultParamsWriter(DefaultParamsWriter):
    def saveImpl(self, path: str) -> None:

        # TODO: delete dataframe like entities from paramMap
        dfParamsMap = {p: value for p, value in self.instance._paramMap.items() if isinstance(value, DataFrame)}
        self.instance._paramMap = {p: value for p, value in self.instance._paramMap.items() if not isinstance(value, DataFrame)}
        # defaultParamMap = self.instance._defaultParamMap

        params_path = os.path.join(path, "params")
        dfs_path = os.path.join(path, "dataframes_params")

        # save what is left
        super().saveImpl(params_path)

        # TODO: save dataframe entities
        for p, df in dfParamsMap.items():
            df_path = os.path.join(dfs_path, p.name)
            # TODO: handle overwriting
            df.write.parquet(df_path)

        self.instance._paramMap = self.instance._paramMap.update(dfParamsMap)

        # TODO: restore dataframe like entities to paramMap