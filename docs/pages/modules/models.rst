Models
=======

.. automodule:: replay.models

RePlay Recommenders
___________________

.. csv-table::
   :header: "Algorithm", "Implementation"
   :widths: 10, 10

    "Popular Recommender", "PySpark"
    "Popular By Queries", "PySpark"
    "Wilson Recommender", "PySpark"
    "Random Recommender", "PySpark"
    "UCB", "PySpark"
    "KL-UCB", "PySpark/Python CPU"
    "Thompson Sampling", "PySpark"
    "K-Nearest Neighbours", "PySpark"
    "Alternating Least Squares", "PySpark"
    "SLIM", "PySpark"
    "Word2Vec Recommender", "PySpark"
    "Association Rules Item-to-Item Recommender", "PySpark"
    "Cluster Recommender", "PySpark"
    "Neural Matrix Factorization (Experimental)", "Python CPU/GPU"
    "MultVAE (Experimental)", "Python CPU/GPU"
    "DDPG (Experimental)", "Python CPU"
    "ADMM SLIM (Experimental)", "Python CPU"
    "Wrapper for implicit (Experimental)", "Python CPU"
    "Wrapper for LightFM (Experimental)", "Python CPU"
    "RL-based CQL Recommender (Experimental)", "PySpark"

To get more info on how to choose base model, please see this  :doc:`page </pages/useful_data/algorithm_selection>`.

Recommender interface
____________________________

.. autoclass:: replay.models.Recommender
    :members:

.. autoclass:: replay.models.base_rec.BaseRecommender
    :members: optimize
    :noindex: optimize

Distributed models
__________________
Models with both training and inference implemented in pyspark.

Popular Recommender
```````````````````
.. autoclass:: replay.models.PopRec

Query Popular Recommender
```````````````````````````
.. autoclass:: replay.models.QueryPopRec

Wilson Recommender
``````````````````
Confidence interval for binomial distribution can be calculated as:

.. math::
    WilsonScore = \frac{\widehat{p}+\frac{z_{ \frac{\alpha}{2}}^{2}}{2n}\pm z_
    {\frac{\alpha}{2}}\sqrt{\frac{\widehat{p}(1-\widehat{p})+\frac{z_
    {\frac{\alpha}{2}}^{2}}{4n}}{n}} }{1+\frac{z_{ \frac{\alpha}{2}}^{2}}{n}}


Where :math:`\hat{p}` -- is an observed fraction of positive ratings.

:math:`z_{\alpha}` 1-alpha quantile of normal distribution.

.. autoclass:: replay.models.Wilson


Random Recommender
``````````````````
.. autoclass:: replay.models.RandomRec
   :special-members: __init__

UCB Recommender
``````````````````
.. autoclass:: replay.models.UCB
   :special-members: __init__

KL-UCB Recommender
``````````````````
.. autoclass:: replay.models.KLUCB
   :special-members: __init__

Thompson Sampling
``````````````````
.. autoclass:: replay.models.ThompsonSampling
   :special-members: __init__

K Nearest Neighbours
````````````````````
.. autoclass:: replay.models.ItemKNN
    :special-members: __init__

.. _als-rec:

Alternating Least Squares
`````````````````````````
.. autoclass:: replay.models.ALSWrap
    :special-members: __init__

Alternating Least Squares on Scala (Experimental)
``````````````````````````````````````````````````````
.. autoclass:: replay.experimental.models.ScalaALSWrap
    :special-members: __init__

SLIM
````
SLIM Recommender calculates similarity between objects to produce recommendations :math:`W`.

Loss function is:

.. math::
    L = \frac 12||A - A W||^2_F + \frac \beta 2 ||W||_F^2+
    \lambda
    ||W||_1

:math:`W` -- item similarity matrix

:math:`A` -- interaction matrix

Finding :math:`W` can be splitted into solving separate linear regressions with ElasticNet regularization.
Thus each row in :math:`W` is optimized with

.. math::
    l = \frac 12||a_j - A w_j||^2_2 + \frac \beta 2 ||w_j||_2^2+
    \lambda ||w_j||_1

To remove trivial solution, we add an extra requirements :math:`w_{jj}=0`,
and :math:`w_{ij}\ge 0`


.. autoclass:: replay.models.SLIM
    :special-members: __init__


Word2Vec Recommender
````````````````````
.. autoclass:: replay.models.Word2VecRec
    :special-members: __init__


Association Rules Item-to-Item Recommender
``````````````````````````````````````````
.. autoclass:: replay.models.AssociationRulesItemRec
    :special-members: __init__
    :members: get_nearest_items


Cluster Recommender
```````````````````
.. autoclass:: replay.models.ClusterRec
    :special-members: __init__


Neural models with distributed inference
________________________________________
Models implemented in pytorch with distributed inference in pyspark.

Neural Matrix Factorization (Experimental)
```````````````````````````````````````````````
.. autoclass:: replay.experimental.models.NeuroMF
    :special-members: __init__

Mult-VAE (Experimental)
````````````````````````````
Variation AutoEncoder

.. image:: /images/vae-gaussian.png

**Problem formulation**

We have a sample of independent equally distributed random values from true distribution
:math:`x_i \sim p_d(x)`, :math:`i = 1, \dots, N`.

Build a probability model :math:`p_\theta(x)` for true distribution :math:`p_d(x)`.

Distribution :math:`p_\theta(x)` allows both to estimate probability density for a given item :math:`x`,
and to sample :math:`x \sim p_\theta(x)`.

**Probability model**

:math:`z \in \mathbb{R}^d` - is a local latent variable, one for each item :math:`x`.

Generative process for variational autoencoder:

1. Sample :math:`z \sim p(z)`.
2. Sample :math:`x \sim p_\theta(x | z)`.

Distribution parameters :math:`p_\theta(x | z)` are defined with neural net weights :math:`\theta`, with input :math:`z`.

Item probability density
:math:`x`:

.. math::
    p_\theta(x) = \mathbb{E}_{z \sim p(z)} p_\theta(x | z)

Use lower estimate bound for the log likelihood.

.. math::
    \log p_\theta(x) = \mathbb{E}_{z \sim q_\phi(z | x)} \log p_\theta(
    x) = \mathbb{E}_{z \sim q_\phi(z | x)} \log \frac{p_\theta(x,
    z) q_\phi(z | x)} {q_\phi(z | x) p_\theta(z | x)} = \\
    = \mathbb{E}_{z
    \sim q_\phi(z | x)} \log \frac{p_\theta(x, z)}{q_\phi(z | x)} + KL(
    q_\phi(z | x) || p_\theta(z | x))

.. math::
    \log p_\theta(x) \geqslant \mathbb{E}_{z \sim q_\phi(z | x)}
    \log \frac{p_\theta(x | z)p(z)}{q_\phi(z | x)} =
    \mathbb{E}_{z \sim q_\phi(z | x)} \log p_\theta(x | z) -
    KL(q_\phi(z | x) || p(z)) = \\
    = L(x; \phi, \theta) \to \max\limits_{\phi, \theta}

:math:`q_\phi(z | x)` is a proposal or a recognition distribution. It is a gaussian with weights :math:`\phi`:
:math:`q_\phi(z | x) = \mathcal{N}(z | \mu_\phi(x), \sigma^2_\phi(x)I)`.

Difference between lower estimate bound :math:`L(x; \phi, \theta)` and log likelihood
:math:`\log p_\theta(x)` - is a KL-divergence between a proposal and aposteriory distribution on :math:`z`:
:math:`KL(q_\phi(z | x) || p_\theta(z | x))`. Maximum value
:math:`L(x; \phi, \theta)` for fixed model parameters
:math:`\theta`
is reached with :math:`q_\phi(z | x) = p_\theta(z | x)`, but explicit calculation of
:math:`p_\theta(z | x)` is not efficient to calculate,
so it is also optimized by :math:`\phi`. The closer :math:`q_\phi(z | x)` to
:math:`p_\theta(z | x)`, the better the estimate.

We usually take normal distribution for :math:`p(z)`:

.. math::
    \varepsilon \sim \mathcal{N}(\varepsilon | 0, I)

.. math::
    z = \mu + \sigma \varepsilon \Rightarrow z \sim \mathcal{N}(z | \mu,
    \sigma^2I)

.. math::
    \frac{\partial}{\partial \phi} L(x; \phi, \theta) = \mathbb{E}_{
    \varepsilon \sim \mathcal{N}(\varepsilon | 0, I)} \frac{\partial}
    {\partial \phi} \log p_\theta(x | \mu_\phi(x) + \sigma_\phi(x)
    \varepsilon) - \frac{\partial}{\partial \phi} KL(q_\phi(z | x) ||
    p(z))

.. math::
    \frac{\partial}{\partial \theta} L(x; \phi, \theta) = \mathbb{E}_{z
    \sim q_\phi(z | x)} \frac{\partial}{\partial \theta} \log
    p_\theta(x | z)

In this case

.. math::
    KL(q_\phi(z | x) || p(z)) = -\frac{1}{2}\sum_{i=1}^{dimZ}(1+
    log(\sigma_i^2) - \mu_i^2-\sigma_i^2)

KL-divergence coefficient can also not be equal to one, in this case:

.. math::
    L(x; \phi, \theta) =
    \mathbb{E}_{z \sim q_\phi(z | x)} \log p_\theta(x | z) -
    \beta \cdot KL(q_\phi(z | x) || p(z)) \to \max\limits_{\phi, \theta}

With :math:`\beta = 0` VAE is the same as the
Denoising AutoEncoder.


.. autoclass:: replay.experimental.models.MultVAE
    :special-members: __init__

DDPG (Experimental)
```````````````````````````
.. autoclass:: replay.experimental.models.DDPG
    :special-members: __init__


CQL Recommender (Experimental)
```````````````````````````````````
Conservative Q-Learning (CQL) algorithm is a SAC-based data-driven deep reinforcement learning algorithm, 
which achieves state-of-the-art performance in offline RL problems.

\* incompatible with python 3.10

.. image:: /images/cql_comparison.png

.. autoclass:: replay.experimental.models.cql.CQL
    :special-members: __init__


Wrappers and other models with distributed inference
____________________________________________________
Wrappers for popular recommendation libraries and algorithms
implemented in python with distributed inference in pyspark.

ADMM SLIM (Experimental)
`````````````````````````````
.. autoclass:: replay.experimental.models.ADMMSLIM
    :special-members: __init__

LightFM (Experimental)
```````````````````````````
.. autoclass:: replay.experimental.models.LightFMWrap
    :special-members: __init__

implicit (Experimental)
````````````````````````````
.. autoclass:: replay.experimental.models.ImplicitWrap
    :special-members: __init__


Neural Networks recommenders
____________________________

Bert4Rec
````````
.. autoclass:: replay.models.nn.Bert4Rec
   :members: __init__, predict_step

SasRec
``````
.. autoclass:: replay.models.nn.SasRec
   :members: __init__, predict_step
