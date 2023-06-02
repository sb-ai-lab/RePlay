import pickle

from replay.scenarios.two_stages.slama_reranker import SlamaWrap
from replay.session_handler import get_spark_session


def main():
    spark = get_spark_session()
    with open("/home/nikolay/wspace/2lvl_train_params.pickle", "rb") as f:
        params = pickle.load(f)

    second_model_config_path = '/home/nikolay/.cache/pypoetry/virtualenvs/' \
                               'replay-rec-Oy-yxr10-py3.9/lib/python3.9/site-packages' \
                               '/sparklightautoml/automl/presets/tabular_config.yml'

    data = spark.read.parquet("/home/nikolay/wspace/2lvl_train_shuffled_0__01.parquet")

    swrap = SlamaWrap(params=params, config_path=second_model_config_path)
    swrap.fit(data)
    results = swrap.predict(data, k=10)
    print(results.count())


if __name__ == "__main__":
    main()
