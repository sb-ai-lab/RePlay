## How to choose a recommender

#### Input Data
- _What is the input?_ 
RePlay models differ by types of data they can process:
    - Collaborative use only user-item interaction logs.
    - Content-based use only user or item features.
    - Hybrid can use both log and features.
- _Are interactions explicit?_
 Our information can be either _explicit_, such as ratings, or _implicit_ such as view, number of play counts. 
\
Some models transfrom any type of data to implicit (_unary ratings_).
- _Will there be new users?_
 Some models need to be completely retrained to give predictions for new users while others don't. 
- _Will there be new items?_
 The same goes for new items.

| Algorithm      | Data         | Interactions | Cold Users | Cold Items |
| ---------------|--------------|-------|-------|-------|
|Popular Recommender        |Collaborative    | converted to unary ratings             | + | - |
|Popular By Users           |Collaborative    | implicit feedback                      | - | - |
|Wilson Recommender         |Collaborative    | binary ratings                         | + | - |
|Random Recommender         |Collaborative    | converted to unary ratings             | + | + |
|K-Nearest Neighbours       |Collaborative    | converted to unary ratings             | + | - |
|Classifier Recommender      |Content-based    | binary ratings                         | + | + |
|Alternating Least Squares  |Collaborative    | implicit feedback                      | - | - |
|Neural Matrix Factorization|Collaborative    | converted to unary ratings             | - | - |
|SLIM                       |Collaborative    | unary ratings, explicit feedback       | + | - |
|ADMM SLIM                  |Collaborative    | unary ratings, explicit feedback       | + | - |
|Mult-VAE                   |Collaborative    | converted to unary ratings             | + | - |
|Word2Vec Recommender       |Collaborative    | converted to unary ratings             | + | - |
|LightFM Wrap               |Hybrid           | [depends on loss](https://making.lyst.com/lightfm/docs/lightfm.html#lightfm)       | + | + |
|Implicit Wrap              |Collaborative    | [depends on model](https://implicit.readthedocs.io/en/latest/index.html)    | - | - |
|Stack Recommender          |Hybrid           | `*`  | `*` | `*` |
|Two Stages Scenario        |Hybrid           | converted to unary ratings for second level    | `*` | `*` |

`*` - depends on base models. 

#### Model requirements
* _Should recommendations be personalized?_ 
* _Should cold users get recommendations?_ (users without interactions).
* _Should model recommend cold items?_ (With no interactions).
* _Should model be able to recommend unseen items?_

| Algorithm      | Personalized | Cold Users | Cold Items |  Recommends Unseen Items |
| ---------------|--------------|-------|-------|-------|
|Popular Recommender          | - | + | - | + |
|Popular By Users             | + | - | - | - |
|Wilson Recommender           | - | + | - | + |
|Random Recommender           | - | + | + | + |
|K-Nearest Neighbours         | + | + | - | + |
|Classifier Recommender       | + | + | + | + |
|Alternating Least Squares    | + | - | - | + |
|Neural Matrix Factorization  | + | - | - | + |
|SLIM                         | + | - | - | + |
|ADMM SLIM                    | + | - | - | + |
|Mult-VAE                     | + | - | - | + |
|Word2Vec Recommender         | + | - | - | + |
|LightFM  Wrap                | + | + | + | + |
|Implicit Wrap                | + | - | - | + |
|Stack Recommender            | + | `*` | `*` | `*` |
|Two Stages Scenario          | + | `*` | `*` | `*` |

`*` - depends on base models. 

More info on [models](../modules/models).
