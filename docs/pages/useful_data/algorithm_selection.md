# How to choose a recommender

## Input Data

### _What is the input?_ 

RePlay models differ by types of data they can process:

- Collaborative use only user-item interaction logs.
- Content-based use only user or item features.
- Hybrid can use both intercations and features.

### _Are interactions explicit?_

 Our information can be either _explicit_ e.g. ratings, or _implicit_ e.g. view, number of play counts. 
\
Some models transform any type of data to implicit (_unary ratings_).

### _Will there be new users?_

Some models need to be completely retrained to give predictions for new users (not present in train data, 
but have interactions history) while others don't. 

### _Will there be new items?_

The same goes for new items.

| Algorithm      | Data         | Interactions | New Users | New Items |
| ---------------|--------------|-------|-------|-------|
|Popular By Users           |Collaborative    | implicit feedback                      | - | - |
|Alternating Least Squares  |Collaborative    | implicit feedback                      | - | - |
|Wilson Recommender         |Collaborative    | binary ratings                         | + | - |
|UCB                        |Collaborative    | binary ratings                         | + | + |
|KL-UCB                     |Collaborative    | binary ratings                         | + | + |
|LinUCB                     |Collaborative    | binary ratings                         | + | - |
|Thompson Sampling          |Collaborative    | binary ratings                         | + | + |
|Conservative Q-Learning (Experimental) | Collaborative | binary ratings             | + | - |
|DDPG (Experimental)        | Collaborative | binary ratings                        | + | - |
|Popular Recommender        |Collaborative    | converted to unary ratings             | + | - |
|Random Recommender         |Collaborative    | converted to unary ratings             | + | + |
|K-Nearest Neighbours       |Collaborative    | converted to unary ratings             | + | - |
|BERT4Rec                   |Collaborative    | converted to unary ratings             | - | - |
|SASRec                   |Collaborative    | converted to unary ratings             | - | - |
|Mult-VAE (Experimental)    |Collaborative    | converted to unary ratings             | + | - |
|Word2Vec Recommender       |Collaborative    | converted to unary ratings             | + | - |
|Association Rules          |Collaborative    | converted to unary ratings             | + | - |
|Neural Matrix Factorization (Experimental)|Collaborative    | converted to unary ratings             | - | - |
|SLIM                       |Collaborative    | unary ratings, explicit feedback       | + | - |
|ADMM SLIM (Experimental)   |Collaborative    | unary ratings, explicit feedback       | + | - |
|ULinUCB (Experimental)     |Hybrid           | binary ratings                         | - | + |
|Neural Thompson Sampling (Experimental)  |Hybrid           | binary ratings                       | + | - |
|Category Popular Recommender |Hybrid           | converted to unary ratings             | + | - |
|Cluster Recommender        |Hybrid           | converted to unary ratings             | + | - |
|LightFM Wrap (Experimental) |Hybrid           | [depends on loss](https://making.lyst.com/lightfm/docs/lightfm.html#lightfm)       | + | + |
|Implicit Wrap (Experimental)|Collaborative    | [depends on model](https://implicit.readthedocs.io/en/latest/index.html)    | - | - |
|Two Stages Scenario (Experimental)|Hybrid           | converted to unary ratings for second level    | `*` | `*` |

`*` - depends on base models. 

Типы взаимодействий:
- __binary ratings__ - модель использует как положительные, так и отрицательные взаимодействия.
- __unary ratings__ - модель учитывает только факт положительного взаимодействия. Все взаимодействия считаются позитивными.
- __implicit feedback__ - неявные оценки.
- __unary ratings, explicit feedback__ - модель способна работать как с унарными оценками (факт взаимодействия), так и с явными оценками (например, рейтингами).
- __depends on loss__ - зависит от функции потерь.

## Model requirements

### _Should recommendations be personalized?_ 
### _Should cold users get recommendations?_ (without any interactions).
### _Should model recommend cold items?_ (that no one interacted with).
### _Should model be able to recommend unseen items?_

| Algorithm      | Personalized | Cold Users | Cold Items |  Recommends Unseen Items |
| ---------------|--------------|-------|-------|-------|
|Popular By Users             | + | - | - | - |
|Alternating Least Squares    | + | - | - | + |
|Wilson Recommender           | - | + | - | + |
|UCB                          | - | + | + | + |
|KL-UCB                       | - | + | + | + |
|LinUCB                       | + | + | - | + |
|Thompson Sampling            | - | + | + | + |
|Conservative Q-Learning (Experimental) | + | + | - | + |
|DDPG (Experimental)          | + | + | - | + |
|Popular Recommender          | - | + | - | + |
|Random Recommender           | - | + | + | + |
|K-Nearest Neighbours         | + | + | - | + |
|BERT4Rec                     | + | - | - | + |
|SASRec                       | + | - | - | + |
|Mult-VAE (Experimental)      | + | - | - | + |
|Word2Vec Recommender         | + | - | - | + |
|Association Rules            | + | - | - | + |
|Neural Matrix Factorization (Experimental) | + | - | - | + |
|SLIM                         | + | - | - | + |
|ADMM SLIM (Experimental)     | + | - | - | + |
|ULinUCB (Experimental)       | + | - | + | + |
|Neural Thompson Sampling (Experimental)| + | + | - | + |
|Category Popular Recommender | - | + | - | + |
|Cluster Recommender          | + | + | - | + |
|LightFM  Wrap (Experimental) | + | + | + | + |
|Implicit Wrap (Experimental) | + | - | - | + |
|Two Stages Scenario (Experimental) | + | `*` | `*` | `*` |

`*` - depends on base models. 

More info on [models](../modules/models).

## Model Comparison
All metrics are calculated at $k=10$
### MovieLens 1m
```{eval-rst}
.. csv-table:: 
   :file: res_1m.csv
   :header-rows: 1
   :stub-columns: 1
```