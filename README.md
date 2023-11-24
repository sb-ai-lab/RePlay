<img src="docs/images/replay_logo_color.svg" height="50"/>
<br>

![GitHub License](https://img.shields.io/github/license/sb-ai-lab/RePlay)
![PyPI - Version](https://img.shields.io/pypi/v/replay-rec)
![PyPI - Downloads](https://img.shields.io/pypi/dm/replay-rec)
<br>
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/sb-ai-lab/RePlay/discussions)

RePlay is an advanced framework designed to facilitate the development and evaluation of recommendation systems. It provides a robust set of tools covering the entire lifecycle of a recommendation system pipeline:

## 🚀 Features:
* **Data Preprocessing and Splitting:** Streamlines the data preparation process for recommendation systems, ensuring optimal data structure and format for efficient processing.
* **Wide Range of Recommendation Models:** Enables building of recommendation models from State-of-the-Art to commonly-used baselines and evaluate their performance and quality.
* **Hyperparameter Optimization:** Offers tools for fine-tuning model parameters to achieve the best possible performance, reducing the complexity of the optimization process.
* **Comprehensive Evaluation Metrics:** Incorporates a wide range of evaluation metrics to assess the accuracy and effectiveness of recommendation models.
* **Model Ensemble and Hybridization:** Supports combining predictions from multiple models and creating two-level (ensemble) models to enhance the quality of recommendations.
* **Seamless Mode Transition:** Facilitates easy transition from offline experimentation to online production environments, ensuring scalability and flexibility.

## 💻 Hardware and Environment Compatibility:
1. **Diverse Hardware Support:** Compatible with various hardware configurations including CPU, GPU, Multi-GPU.
2. **Cluster Computing Integration:** Integrating with PySpark for distributed computing, enabling scalability for large-scale recommendation systems.

## 📖 Documentation is available [here](https://sb-ai-lab.github.io/RePlay/).

<a name="toc"></a>
# Table of Contents

* [Installation](#installation)
* [Quickstart](#quickstart)
* [Resources](#examples)
* [Contributing to RePlay](#contributing)


<a name="installation"></a>
## 🔧 Installation

Installation via `pip` package manager is recommended by default:

```bash
pip install replay-rec
```

In this case it will be installed the `core` package without `PySpark` and `PyTorch` dependencies.
Also `experimental` submodule will not be installed.

To install `experimental` submodule please specify the version with `rc0` suffix.
For example:

```bash
pip install replay-rec==XX.YY.ZZrc0
```

### Extras

In addition to the core package, several extras are also provided, including:
- `[spark]`: Install PySpark functionality
- `[torch]`: Install PyTorch and Lightning functionality
- `[all]`: `[spark]` `[torch]`

Example:
```bash
# Install core package with PySpark dependency
pip install replay-rec[spark]

# Install package with experimental submodule and PySpark dependency
pip install replay-rec[spark]==XX.YY.ZZrc0
```

To build RePlay from sources please use the [instruction](CONTRIBUTING.md#installing-from-the-source).

If you encounter an error during RePlay installation, check the [troubleshooting](https://sb-ai-lab.github.io/RePlay/pages/installation.html#troubleshooting) guide.


<a name="quickstart"></a>
## 📈 Quickstart

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
## 📑  Resources

### Usage examples
1. [01_replay_basics.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/examples/01_replay_basics.ipynb) - get started with RePlay.
2. [02_models_comparison.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/examples/02_models_comparison.ipynb) - reproducible models comparison on [MovieLens-1M dataset](https://grouplens.org/datasets/movielens/1m/).
3. [03_features_preprocessing_and_lightFM.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/examples/03_features_preprocessing_and_lightFM.ipynb) - LightFM example with pyspark for feature preprocessing.
4. [04_splitters.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/examples/04_splitters.ipynb) - An example of using RePlay data splitters.
5. [05_feature_generators.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/examples/05_feature_generators.ipynb) - Feature generation with RePlay.
6. [06_item2item_recommendations.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/examples/06_item2item_recommendations.ipynb) - Item to Item recommendations example.
7. [07_filters.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/examples/07_filters.ipynb) - An example of using filters.
8. [08_recommending_for_categories.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/examples/08_recommending_for_categories.ipynb) - An example of recommendation for product categories.
9. [09_sasrec_example.ipynb](https://github.com/sb-ai-lab/RePlay/blob/main/examples/09_sasrec_example.ipynb) - An example of using transformers to generate recommendations.


### Videos and papers
* **Video guides**:
	- [Replay for offline recommendations, AI Journey 2021](https://www.youtube.com/watch?v=ejQZKGAG0xs)

* **Research papers**:
	- Yan-Martin Tamm, Rinchin Damdinov, Alexey Vasilev [Quality Metrics in Recommender Systems: Do We Calculate Metrics Consistently?](https://dl.acm.org/doi/10.1145/3460231.3478848)

<a name="contributing"></a>
## 💡 Contributing to RePlay

We welcome community contributions. For details please check our [contributing guidelines](CONTRIBUTING.md).
