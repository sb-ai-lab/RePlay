import os

from pyspark.ml.util import DefaultParamsWriter
from pyspark.sql import DataFrame


class DataframeAwareDefaultParamsWriter(DefaultParamsWriter):
    def saveImpl(self, path: str) -> None:
        dfParamsMap = {p: value for p, value in self.instance._paramMap.items() if isinstance(value, DataFrame)}
        self.instance._paramMap = {p: value for p, value in self.instance._paramMap.items() if not isinstance(value, DataFrame)}
        dfDefaultParamMap = {p: value for p, value in self.instance._defaultParamMap.items() if isinstance(value, DataFrame)}
        self.instance._defaultParamMap = {p: value for p, value in self.instance._defaultParamMap.items() if not isinstance(value, DataFrame)}

        dfs_path = os.path.join(path, "dataframes_params")
        default_dfs_path = os.path.join(path, "dataframes_default_params")

        # save what is left
        super().saveImpl(path)

        for p, df in dfParamsMap.items():
            df_path = os.path.join(dfs_path, p.name)
            df.write.parquet(df_path)

        for p, df in dfDefaultParamMap.items():
            df_path = os.path.join(default_dfs_path, p.name)
            df.write.parquet(df_path)

        self.instance._paramMap = self.instance._paramMap.update(dfParamsMap)
        self.instance._defaultParamMap = self.instance._defaultParamMap.update(dfDefaultParamMap)
