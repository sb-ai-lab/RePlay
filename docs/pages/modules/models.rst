Модели
=======

.. automodule:: replay.models

**Базовые алгоритмы**

.. csv-table::
   :header: "Алгоритм", "Реализация", "Описание"
   :widths: 10, 10, 10

    "Popular Recommender", "PySpark", "Рекомендует популярные объекты (встречавшиеся в истории взаимодействия чаще остальных)"
    "Popular By Users", "PySpark", "Рекомендует объекты, которые пользователь ранее выбирал чаще всего"
    "Wilson Recommender", "Python CPU", "Рекомендует объекты с лучшими оценками. Оценка объекта определяется как нижняя граница доверительного интервала Уилсона для доли положительных взаимодействий"
    "Random Recommender", "PySpark", "Рекомендует случайные объекты или сэмплирует с вероятностью, пропорциональной популярности объекта"
    "K-Nearest Neighbours", "PySpark", "Рекомендует объекты, похожие на те, с которыми взаимодействовал пользователь"
    "Classifier Recommender", "PySpark", "Алгоритм бинарной классификации для релевантности объекта для пользователя по их признакам"
    "Alternating Least Squares", "PySpark", "Алгоритм матричной факторизации `Collaborative Filtering for Implicit Feedback Datasets <https://ieeexplore.ieee.org/document/4781121>`_"
    "Neural Matrix Factorization", "Python CPU/GPU", "Алгоритм нейросетевой матричной факторизации на базе `Neural Collaborative Filtering <https://arxiv.org/pdf/1708.05031.pdf>`_"
    "SLIM", "PySpark", "Алгоритм, обучающий матрицу близости объектов для восстановления матрицы взаимодействия `SLIM: Sparse Linear Methods for Top-N Recommender Systems <http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf>`_"
    "ADMM SLIM", "Python CPU", "Улучшение стандартного алгоритма SLIM, `ADMM SLIM: Sparse Recommendations for Many Users <http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf>`_"
    "MultVAE", "Python CPU/GPU", "Вариационный автоэнкодер, восстанавливающий вектор взаимодействий для пользователя `Variational Autoencoders for Collaborative Filtering <https://arxiv.org/pdf/1802.05814.pdf>`_"
    "Word2Vec Recommender", "Python CPU/GPU", "Рекомендатель на основе word2vec, в котором объекты сопоставляются словам, а пользователи - предложениям"
    "Обертка LightFM", "Python CPU", "Обертка для обучения `моделей LightFM <https://making.lyst.com/lightfm/docs/home.html>`_"
    "Обертка Implicit", "Python CPU", "Обертка для обучения `моделей Implicit <https://implicit.readthedocs.io/en/latest/>`_"

Для всех базовых алгоритмов выдача рекоментаций (inference) реализована с использованием PySpark.

**Многоуровневые алгоритмы**

.. csv-table::
   :header: "Алгоритм", "Реализация", "Описание"
   :widths: 10, 10, 10

   "Stack Recommender", "`*`", "Модель стекинга, перевзвешивающая предсказания моделей первого уровня"
   "Двухуровневый классификатор", "`*`", "Классификатор, использующий для обучения эмбеддинги пользователей и объектов, полученные базовым алгоритмом (например, матричной факторизацией), и признаки пользователей и объектов, переданные пользователем"

`*` - зависит от алгоритмов, используемых в качестве базовых.

Больше информации об алгоритмах и их применимости для различных данных :doc:`здесь </pages/useful_data/algorithm_selection>`.

.. autoclass:: replay.models.Recommender
    :members:

.. _pop-rec:

Popular Recommender
--------------------

.. autoclass:: replay.models.PopRec

Wilson Recommender
-------------------

Доверительный интервал для биномиального распределения можно высчитать по следующей формуле:

.. math::
    WilsonScore = \frac{\widehat{p}+\frac{z_{ \frac{\alpha}{2}}^{2}}{2n}\pm z_
    {\frac{\alpha}{2}}\sqrt{\frac{\widehat{p}(1-\widehat{p})+\frac{z_
    {\frac{\alpha}{2}}^{2}}{4n}}{n}} }{1+\frac{z_{ \frac{\alpha}{2}}^{2}}{n}}


Где :math:`\hat{p}` -- наблюдаемая доля положительных оценок (1 по отношению к 0).

:math:`z_{\alpha}` 1-альфа квантиль нормального распределения.

.. autoclass:: replay.models.Wilson

Random Recommender
------------------

.. autoclass:: replay.models.RandomRec
   :special-members: __init__

.. _knn-model:

K Nearest Neighbours
----------------------

.. autoclass:: replay.models.KNN
    :special-members: __init__

Classifier Recommender
----------------------

..  autoclass:: replay.models.ClassifierRec
    :special-members: __init__

.. _als-rec:

Alternating Least Squares
---------------------------

.. autoclass:: replay.models.ALSWrap
    :special-members: __init__

Neural Matrix Factorization
-----------------------------

.. autoclass:: replay.models.NeuroMF
    :special-members: __init__

SLIM
--------

SLIM Recommender основан на обучении матрицы близости объектов
:math:`W`.

Оптимизируется следующий функционал:

.. math::
    L = \frac 12||A - A W||^2_F + \frac \beta 2 ||W||_F^2+
    \lambda
    ||W||_1

:math:`W` -- матрица близости между объектами

:math:`A` -- матрица взаимодействия пользователей/объектов

Задачу нахождения матрицы :math:`W` можно разбить на множество
задач линейной регрессии с регуляризацией ElasticNet. Таким образом,
для каждой строки матрицы :math:`W` необходимо оптимизировать следующий
функционал

.. math::
    l = \frac 12||a_j - A w_j||^2_2 + \frac \beta 2 ||w_j||_2^2+
    \lambda ||w_j||_1

Чтобы решение было не тривиальным, его ищут с ограничением :math:`w_{jj}=0`,
кроме этого :math:`w_{ij}\ge 0`


.. autoclass:: replay.models.SLIM
    :special-members: __init__

ADMM SLIM
----------

.. autoclass:: replay.models.ADMMSLIM
    :special-members: __init__

LightFM
-----------

.. autoclass:: replay.models.LightFMWrap
    :special-members: __init__

implicit
---------

.. autoclass:: replay.models.ImplicitWrap
    :special-members: __init__


Mult-VAE
--------

Вариационный автокодировщик. Общая схема его работы
представлена на рисунке.

.. image:: /images/vae-gaussian.png

**Постановка задачи**

Дана выборка независимых одинаково распределенных величин из истинного
распределения :math:`x_i \sim p_d(x)`, :math:`i = 1, \dots, N`.

Задача - построить вероятностную модель :math:`p_\theta(x)` истинного
распределения :math:`p_d(x)`.

Распределение :math:`p_\theta(x)` должно позволять как оценить плотность
вероятности для данного объекта :math:`x`, так и сэмплировать
:math:`x \sim p_\theta(x)`.

**Вероятностная модель**

:math:`z \in \mathbb{R}^d` - локальная латентная переменная, т. е. своя для
каждого объекта :math:`x`.

Генеративный процесс вариационного автокодировщика:

1. Сэмплируем :math:`z \sim p(z)`.
2. Сэмплируем :math:`x \sim p_\theta(x | z)`.

Параметры распределения :math:`p_\theta(x | z)` задаются нейросетью с
весами :math:`\theta`, получающей на вход вектор :math:`z`.

Индуцированная генеративным процессом плотность вероятности объекта
:math:`x`:

.. math::
    p_\theta(x) = \mathbb{E}_{z \sim p(z)} p_\theta(x | z)

В случачае ВАЕ для максимизации правдоподобия максимизируют вариационную
нижнюю оценку на логарифм правдоподобия

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

:math:`q_\phi(z | x)` называется предложным (proposal) или распознающим
(recognition) распределением. Это гауссиана, чьи параметры задаются
нейросетью с весами :math:`\phi`:
:math:`q_\phi(z | x) = \mathcal{N}(z | \mu_\phi(x), \sigma^2_\phi(x)I)`.

Зазор между вариационной нижней оценкой :math:`L(x; \phi, \theta)` на
логарифм правдоподобия модели и самим логарифмом правдоподобия
:math:`\log p_\theta(x)` - это KL-дивергенция между предолжным и
апостериорным распределением на :math:`z`:
:math:`KL(q_\phi(z | x) || p_\theta(z | x))`. Максимальное значение
:math:`L(x; \phi, \theta)` при фиксированных параметрах модели
:math:`\theta`
достигается при :math:`q_\phi(z | x) = p_\theta(z | x)`, но явное
вычисление :math:`p_\theta(z | x)` требует слишком большого числа
ресурсов, поэтому вместо этого вычисления вариационная нижняя оценка
оптимизируется также по :math:`\phi`. Чем ближе :math:`q_\phi(z | x)` к
:math:`p_\theta(z | x)`, тем точнее вариационная нижняя оценка.

Обычно в качестве априорного распределения :math:`p(z)` используетя
какое-то простое распределение, чаще всего нормальное:

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

В этом случае

.. math::
    KL(q_\phi(z | x) || p(z)) = -\frac{1}{2}\sum_{i=1}^{dimZ}(1+
    log(\sigma_i^2) - \mu_i^2-\sigma_i^2)

Также коэффициент при KL-дивергенции (коэффициент отжига) может быть
положен не равным единице. Тогда оптимизируемая функция выглядит
следующим образом

.. math::
    L(x; \phi, \theta) =
    \mathbb{E}_{z \sim q_\phi(z | x)} \log p_\theta(x | z) -
    \beta \cdot KL(q_\phi(z | x) || p(z)) \to \max\limits_{\phi, \theta}

При :math:`\beta = 0` VAE (вариационный автокодировщик) превращается в
DAE (шумоподавляющий автокодировщик)


.. autoclass:: replay.models.MultVAE
    :special-members: __init__

Word2Vec Recommender
--------------------

.. autoclass:: replay.models.Word2VecRec
    :special-members: __init__

Stack
-------

.. autoclass:: replay.models.Stack
    :special-members: __init__