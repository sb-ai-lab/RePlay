# RePlay

RePlay is a library providing tools for all stages of creating a recommendation system, from data preprocessing to model evaluation and comparison.

RePlay uses PySpark to handle big data.

You can

- Filter and split data
- Train models
- Optimize hyper parameters
- Evaluate predictions with metrics
- Combine predictions from different models
- Create a two-level model

Documentation is available [here](https://sb-ai-lab.github.io/RePlay/).

<a name="toc"></a>
# Table of Contents

* [Installation](#installation)
* [Quickstart](#quickstart)
* [Resources](#examples)
* [Contributing to RePlay](#contributing)


<a name="installation"></a>
## Installation

Use Linux machine with Python 3.7-3.9, Java 8+ and C++ compiler.

```bash
pip install replay-rec
```

To get the latest development version or RePlay, [install it from the GitHab repository](https://sb-ai-lab.github.io/RePlay/pages/installation.html#development).
It is preferable to use a virtual environment for your installation.

If you encounter an error during RePlay installation, check the [troubleshooting](https://sb-ai-lab.github.io/RePlay/pages/installation.html#troubleshooting) guide.


<a name="quickstart"></a>
## Quickstart

```python
from rs_datasets import MovieLens

from replay.preprocessing.data_preparator import DataPreparator, Indexer
from replay.metrics import HitRate, NDCG
from replay.models import ItemKNN
from replay.utils.session_handler import State
from replay.splitters import UserSplitter

spark = State().session

ml_1m = MovieLens("1m")

# data preprocessing
preparator = DataPreparator()
log = preparator.transform(
    columns_mapping={
        'user_id': 'user_id',
        'item_id': 'item_id',
        'relevance': 'rating',
        'timestamp': 'timestamp'
    }, 
    data=ml_1m.ratings
)
indexer = Indexer(user_col='user_id', item_col='item_id')
indexer.fit(users=log.select('user_id'), items=log.select('item_id'))
log_replay = indexer.transform(df=log)

# data splitting
user_splitter = UserSplitter(
    item_test_size=10,
    user_test_size=500,
    drop_cold_items=True,
    drop_cold_users=True,
    shuffle=True,
    seed=42,
)
train, test = user_splitter.split(log_replay)

# model training
model = ItemKNN()
model.fit(train)

# model inference
recs = model.predict(
    log=train,
    k=K,
    users=test.select('user_idx').distinct(),
    filter_seen_items=True,
)

# model evaluation
metrics = Experiment(test,  {NDCG(): K, HitRate(): K})
metrics.add_result("knn", recs)
```

<a name="examples"></a>
## Resources

### Usage examples
1. [01_replay_basics.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/experiments/01_replay_basics.ipynb) - get started with RePlay.
2. [02_models_comparison.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/experiments/02_models_comparison.ipynb) - reproducible models comparison on [MovieLens-1M dataset](https://grouplens.org/datasets/movielens/1m/).
3. [03_features_preprocessing_and_lightFM.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/experiments/03_features_preprocessing_and_lightFM.ipynb) - LightFM example with pyspark for feature preprocessing.
3. [04_splitters.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/experiments/04_splitters.ipynb) - An example of using RePlay data splitters.
3. [05_feature_generators.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/experiments/05_feature_generators.ipynb) - Feature generation with RePlay.


### Videos and papers
* **Video guides**:
	- [Replay for offline recommendations, AI Journey 2021](https://www.youtube.com/watch?v=ejQZKGAG0xs)

* **Research papers**:
	- Yan-Martin Tamm, Rinchin Damdinov, Alexey Vasilev [Quality Metrics in Recommender Systems: Do We Calculate Metrics Consistently?](https://dl.acm.org/doi/10.1145/3460231.3478848)

<a name="contributing"></a>
## Contributing to RePlay

We welcome community contributions. For details please check our [contributing guidelines](CONTRIBUTING.md).
