.. _splitters:

Сплиттеры
==========

.. automodule:: replay.splitters

Все модули возвращают сплиты по методу ``split``.

.. autofunction:: replay.splitters.base_splitter.Splitter.split

Делим внутри пользователя
--------------------------

.. automodule:: replay.splitters.user_log_splitter
   :members: UserSplitter
   :special-members: __init__

Для разделения внутри пользователя также доступно разделение по фолдам.

.. autofunction:: replay.splitters.user_log_splitter.k_folds

Делим весь лог
------------------

.. automodule:: replay.splitters.log_splitter
   :members: DateSplitter, RandomSplitter, ColdUserByDateSplitter, ColdUserRandomSplitter
   :special-members:
