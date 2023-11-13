# RePlay

RePlay is a library providing tools for all stages of creating a recommendation system, from data preprocessing to model evaluation and comparison.

RePlay can use PySpark to handle big data.

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

Installation via `pip` package manager is recommended by default:

```bash
pip install replay-rec
```

By default `experimental` submodule is not installed, if you need this functionality please specify the version with `rc0` suffix.

For example:

```bash
pip install replay-rec==XX.YY.ZZrc0
```

To build RePlay from sources please use the [instruction](CONTRIBUTING.md#installing-from-the-source).

If you encounter an error during RePlay installation, check the [troubleshooting](https://sb-ai-lab.github.io/RePlay/pages/installation.html#troubleshooting) guide.


<a name="quickstart"></a>
## Quickstart

```python
from rs_datasets import MovieLens

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.data.dataset_utils import DatasetLabelEncoder
from replay.metrics import HitRate, NDCG, Experiment
from replay.models import ItemKNN
from replay.utils import convert2spark
from replay.utils.session_handler import State
from replay.splitters import RatioSplitter

spark = State().session

ml_1m = MovieLens("1m")
K=10

# data preprocessing
interactions = convert2spark(ml_1m.ratings)

# data splitting
splitter = RatioSplitter(
    test_size=0.3,
    divide_column="user_id",
    query_column="user_id",
    item_column="item_id",
    timestamp_column="timestamp",
    drop_cold_items=True,
    drop_cold_users=True,
)
train, test = splitter.split(interactions)

# dataset creating
feature_schema = FeatureSchema(
    [
        FeatureInfo(
            column="user_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.QUERY_ID,
        ),
        FeatureInfo(
            column="item_id",
            feature_type=FeatureType.CATEGORICAL,
            feature_hint=FeatureHint.ITEM_ID,
        ),
        FeatureInfo(
            column="rating",
            feature_type=FeatureType.NUMERICAL,
            feature_hint=FeatureHint.RATING,
        ),
        FeatureInfo(
            column="timestamp",
            feature_type=FeatureType.NUMERICAL,
            feature_hint=FeatureHint.TIMESTAMP,
        ),
    ]
)

train_dataset = Dataset(
    feature_schema=feature_schema,
    interactions=train,
)
test_dataset = Dataset(
    feature_schema=feature_schema,
    interactions=test,
)

# data encoding
encoder = DatasetLabelEncoder()
train_dataset = encoder.fit_transform(train_dataset)
test_dataset = encoder.transform(test_dataset)

# model training
model = ItemKNN()
model.fit(train_dataset)

# model inference
encoded_recs = model.predict(
    dataset=train_dataset,
    k=K,
    queries=test_dataset.query_ids,
    filter_seen_items=True,
)

recs = encoder.query_and_item_id_encoder.inverse_transform(encoded_recs)

# model evaluation
metrics = Experiment(
    [NDCG(K), HitRate(K)],
    test,
    query_column="user_id",
    item_column="item_id",
    rating_column="rating",
)
metrics.add_result("ItemKNN", recs)
print(metrics.results)
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
