Сравнение моделей
==================

Модели, доступные в библиотеке были замерены при одинаковых условиях и имеют следующие результаты:

Если для модели уже известны оптимальные параметры, замерить качество можно по аналогии с примером:

.. code-block:: python

   from rs_datasets import MovieLens

   from sponge_bob_magic.metrics import *
   from sponge_bob_magic.models.ease import EASE
   from sponge_bob_magic.models import RandomPop
   from sponge_bob_magic.splitters import UserSplitter
   from sponge_bob_magic.experiment import Experiment

   seed = 1337
   k = [10, 20, 50]
   ml = MovieLens("20m")
   data = ml.ratings
   splitter = UserSplitter(max(k), seed=seed)
   train, test = splitter.split(data)

   e = Experiment(test,
                  [NDCG(), HitRate(), Coverage(data), MAP(), MRR(),
                   Precision(), Recall(), RocAuc(), Surprisal(data),
                   Unexpectedness(data, RandomPop(data, seed))],
                  k
                  )

   model = EASE()
   pred = model.fit_predict(train, max(k))
   e.add_result("my_model", pred)


