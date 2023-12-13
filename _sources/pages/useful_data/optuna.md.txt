# How to optimize a model

This page will explore different ways to optimize models in RePlay.
 
First we initialize a model

```python
from replay.models import SLIM
model = SLIM()
```

If you just want to optimize a model with default settings, all you have to specify is a data to use for optimization.

```python
model.optimize(train, val)
```
This will return a dict with the best parameters and set them. 

If you are not pleased with the results you can continue optimizing by calling `optimize` with `new_study=False`, 
it will continue optimizing right where it stopped. Optuna study is stored as a model attribute. 
For example, you can see all trials with `model.study.trials`.


```{eval-rst}
.. autofunction:: replay.models.base_rec.BaseRecommender.optimize
```

You can either use default borders or specify them yourself. 
A list of searchable parameters is specified in `_search_space` attribute.

```python
model._search_space

{'beta': {'type': 'loguniform', 'args': [1e-06, 5]},
 'lambda_': {'type': 'loguniform', 'args': [1e-06, 2]}}
```

If you specify only one of the parameters, the other one will not be optimized.

```python
model = SLIM(lambda_=1)
model.optimize(train, val, param_borders={'beta': [0.1, 1]})
```

