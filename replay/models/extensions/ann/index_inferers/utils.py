import pandas as pd
from scipy.sparse import csr_matrix


def get_csr_matrix(
    user_idx: pd.Series,
    vector_items: pd.Series,
    vector_ratings: pd.Series,
) -> csr_matrix:
    """Creates and returns csr matrix of user-item interactions"""
    return csr_matrix(
        (
            vector_ratings.explode().values.astype(float),
            (
                user_idx.repeat(vector_items.apply(lambda x: len(x))).values,
                vector_items.explode().values.astype(int),
            ),
        ),
        shape=(
            user_idx.max() + 1,
            vector_items.apply(lambda x: max(x)).max() + 1,
        ),
    )
