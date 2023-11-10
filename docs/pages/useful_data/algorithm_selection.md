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
|Popular Recommender        |Collaborative    | converted to unary ratings             | + | - |
|Popular By Users           |Collaborative    | implicit feedback                      | - | - |
|Wilson Recommender         |Collaborative    | binary ratings                         | + | - |
|UCB                        |Collaborative    | binary ratings                         | + | + |
|Random Recommender         |Collaborative    | converted to unary ratings             | + | + |
|K-Nearest Neighbours       |Collaborative    | converted to unary ratings             | + | - |
|Alternating Least Squares  |Collaborative    | implicit feedback                      | - | - |
|Neural Matrix Factorization (Experimental)|Collaborative    | converted to unary ratings             | - | - |
|SLIM                       |Collaborative    | unary ratings, explicit feedback       | + | - |
|ADMM SLIM (Experimental)   |Collaborative    | unary ratings, explicit feedback       | + | - |
|Mult-VAE (Experimental)    |Collaborative    | converted to unary ratings             | + | - |
|Word2Vec Recommender       |Collaborative    | converted to unary ratings             | + | - |
|Association Rules          |Collaborative    | converted to unary ratings             | + | - |
|Cluster Recommender        |Hybrid           | converted to unary ratings             | + | - |
|LightFM Wrap (Experimental) |Hybrid           | [depends on loss](https://making.lyst.com/lightfm/docs/lightfm.html#lightfm)       | + | + |
|Implicit Wrap (Experimental)|Collaborative    | [depends on model](https://implicit.readthedocs.io/en/latest/index.html)    | - | - |
|Two Stages Scenario (Experimental)|Hybrid           | converted to unary ratings for second level    | `*` | `*` |

`*` - depends on base models. 

## Model requirements

### _Should recommendations be personalized?_ 
### _Should cold users get recommendations?_ (without any interactions).
### _Should model recommend cold items?_ (that no one interacted with).
### _Should model be able to recommend unseen items?_

| Algorithm      | Personalized | Cold Users | Cold Items |  Recommends Unseen Items |
| ---------------|--------------|-------|-------|-------|
|Popular Recommender          | - | + | - | + |
|Popular By Users             | + | - | - | - |
|Wilson Recommender           | - | + | - | + |
|UCB                          | - | + | + | + |
|Random Recommender           | - | + | + | + |
|K-Nearest Neighbours         | + | + | - | + |
|Alternating Least Squares    | + | - | - | + |
|Neural Matrix Factorization (Experimental) | + | - | - | + |
|SLIM                         | + | - | - | + |
|ADMM SLIM (Experimental)     | + | - | - | + |
|Mult-VAE (Experimental)      | + | - | - | + |
|Word2Vec Recommender         | + | - | - | + |
|Association Rules            | + | - | - | + |
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