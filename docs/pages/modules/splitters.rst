.. _splitters:

Сплиттеры
==========

.. automodule:: replay.splitters

Все модули возвращают сплиты по методу ``split``.

.. autofunction:: replay.splitters.base_splitter.Splitter.split


UserSplitter
-------------

.. autoclass:: replay.splitters.user_log_splitter.UserSplitter
   :special-members: __init__

k_folds
---------

Для разделения внутри пользователя также доступно разделение по фолдам.

.. autofunction:: replay.splitters.user_log_splitter.k_folds


DateSplitter
-------------
.. autoclass:: replay.splitters.log_splitter.DateSplitter
   :special-members: __init__

RandomSplitter
----------------
.. autoclass:: replay.splitters.log_splitter.RandomSplitter
   :special-members: __init__

NewUsersSplitter
-----------------
.. autoclass:: replay.splitters.log_splitter.NewUsersSplitter
   :special-members: __init__

ColdUserRandomSplitter
------------------------
.. autoclass:: replay.splitters.log_splitter.ColdUserRandomSplitter
   :special-members: __init__
