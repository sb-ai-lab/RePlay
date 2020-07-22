Сравнение моделей
==================

Модели, доступные в библиотеке были замерены при одинаковых условиях и имеют следующие результаты:

Если для модели уже известны оптимальные параметры, замерить качество можно по аналогии с примером:

.. code-block:: python

   from rs_datasets import MovieLens

   from replay.metrics import *
   from replay.models.ease import EASE
   from replay.models import RandomPop
   from replay.splitters import UserSplitter
   from replay.experiment import Experiment

   seed = 1337
   k = [10, 20, 50]
   ml = MovieLens("20m")
   data = ml.ratings
   splitter = UserSplitter(max(k), seed=seed)
   train, test = splitter.split(data)

   e = Experiment(test,
                  {NDCG(): k, HitRate(): k, Coverage(data): k, MAP(): k, MRR(): k,
                   Precision(): k, Recall(): k, RocAuc(): k, Surprisal(data): k,
                   Unexpectedness(data, RandomPop(data, seed)): k}
                  )

   model = EASE()
   pred = model.fit_predict(train, max(k))
   e.add_result("my_model", pred)


Пример подбора гиперпараметров для моделей из implicit с помощью nevergrad и сравнения с бейзлайном
можно посмотреть в ``experiments/tune_and_compare_ng.py``.