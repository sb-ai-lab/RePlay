import logging
import time
import warnings

import pandas as pd
import torch
import tqdm
from optuna.exceptions import ExperimentalWarning
from pyspark.sql import functions as sf

from replay.data_preparator import DataPreparator, Indexer
from replay.experiment import Experiment
from replay.metrics import HitRate, NDCG, MAP, MRR, Coverage, Surprisal
from replay.models import ALSWrap, KNN, LightFMWrap, SLIM, UCB, CQL, Wilson
from replay.session_handler import State
from replay.splitters import DateSplitter
from replay.utils import get_log_info


def fit_predict_add_res(name, model, experiment, train, top_k, test_users):
    """
    Run fit_predict for the `model`, measure time on fit_predict and evaluate metrics
    """
    start_time = time.time()

    model.fit(log=train)
    fit_time = time.time() - start_time

    pred = model.predict(log=train, k=top_k, users=test_users)
    pred.cache()
    pred.count()
    predict_time = time.time() - start_time - fit_time

    experiment.add_result(name, pred)
    metric_time = time.time() - start_time - fit_time - predict_time

    experiment.results.loc[name, 'fit_time'] = fit_time
    experiment.results.loc[name, 'predict_time'] = predict_time
    experiment.results.loc[name, 'metric_time'] = metric_time
    experiment.results.loc[name, 'full_time'] = (fit_time + predict_time + metric_time)
    pred.unpersist()


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    use_gpu = torch.cuda.is_available()

    spark = State().session
    spark.sparkContext.setLogLevel('ERROR')

    prefix = "./data/"
    df_log = pd.read_csv(
        f'{prefix}/ml1m_ratings.dat', sep='\t',
        names=['user_id', 'item_id', 'relevance', 'timestamp']
    )

    col_mapping = {key: key for key in ['user_id', 'item_id', 'relevance', 'timestamp']}

    data_preparator = DataPreparator()
    log = data_preparator.transform(columns_mapping=col_mapping, data=df_log)

    K = 10
    K_list_metrics = [1, 5, 10]
    SEED = 12345

    indexer = Indexer()
    indexer.fit(users=log.select('user_id'), items=log.select('item_id'))

    # will consider ratings >= 3 as positive feedback.
    # A positive feedback is treated with relevance = 1
    only_positives_log = log.filter(sf.col('relevance') >= 3).withColumn('relevance', sf.lit(1.))
    # negative feedback will be used for Wilson and UCB models
    only_negatives_log = log.filter(sf.col('relevance') < 3).withColumn('relevance', sf.lit(0.))

    pos_log = indexer.transform(df=only_positives_log)

    # train/test split
    date_splitter = DateSplitter(
        test_start=0.98,
        drop_cold_items=True,
        drop_cold_users=True,

    )
    train, test = date_splitter.split(pos_log)
    train.cache(), test.cache()
    print('train info:\n', get_log_info(train))
    print('test info:\n', get_log_info(test))

    test_start = test.agg(sf.min('timestamp')).collect()[0][0]

    # train with both positive and negative feedback
    pos_neg_train = (
        train
        .withColumn('relevance', sf.lit(1.))
        .union(
            indexer.transform(
                only_negatives_log.filter(sf.col('timestamp') < test_start)
            )
        )
    )
    pos_neg_train.cache()
    pos_neg_train.count()

    experiment = Experiment(test, {
        MAP(): K, NDCG(): K, HitRate(): K_list_metrics, Coverage(train): K, Surprisal(train): K,
        MRR(): K
    })

    algorithms = {
        'CQL': CQL(use_gpu=use_gpu, k=K, n_epochs=3),
        'ALS': ALSWrap(seed=SEED),
        'KNN': KNN(num_neighbours=K),
        'LightFM': LightFMWrap(random_state=SEED),
        'SLIM': SLIM(seed=SEED),
        'UCB': UCB(exploration_coef=0.5)
    }

    logger = logging.getLogger("replay")
    test_users = test.select('user_idx').distinct()

    for name in tqdm.tqdm(algorithms.keys(), desc='Model'):
        model = algorithms[name]

        logger.info(msg='{} started'.format(name))

        train_ = train
        if isinstance(model, (Wilson, UCB)):
            train_ = pos_neg_train
        fit_predict_add_res(name, model, experiment, train=train_, top_k=K, test_users=test_users)
        print(
            experiment.results[
                ['NDCG@{}'.format(K), 'MRR@{}'.format(K), 'Coverage@{}'.format(K), 'fit_time']
            ].sort_values('NDCG@{}'.format(K), ascending=False)
        )


