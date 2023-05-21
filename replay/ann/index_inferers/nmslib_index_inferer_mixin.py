import pandas as pd
from scipy.sparse import csr_matrix


class NmslibIndexInfererMixin:
    @staticmethod
    def get_csr_matrix(
        user_idx: pd.Series,
        vector_items: pd.Series,
        vector_relevances: pd.Series,
    ) -> csr_matrix:
        return csr_matrix(
            (
                vector_relevances.explode().values.astype(float),
                (
                    user_idx.repeat(
                        vector_items.apply(
                            lambda x: len(x)
                        )  # pylint: disable=unnecessary-lambda
                    ).values,
                    vector_items.explode().values.astype(int),
                ),
            ),
            shape=(
                user_idx.max() + 1,
                vector_items.apply(lambda x: max(x)).max()
                + 1,  # pylint: disable=unnecessary-lambda
            ),
        )
