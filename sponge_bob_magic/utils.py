from pyspark.sql import DataFrame


def get_distinct_values_in_column(df: DataFrame, column: str):
    return set([row[column]
                for row in (df
                            .select(column)
                            .distinct()
                            .collect())
                ])