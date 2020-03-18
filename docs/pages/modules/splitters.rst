.. _splitters:

Сплиттеры
==========

.. automodule:: sponge_bob_magic.splitters

Все модули возвращают сплиты по методу ``split``.

.. autofunction:: sponge_bob_magic.splitters.base_splitter.Splitter.split

Делим внутри пользователя
--------------------------

.. automodule:: sponge_bob_magic.splitters.user_log_splitter
   :members: UserSplitter
   :special-members: __init__

Делим весь лог
------------------

.. automodule:: sponge_bob_magic.splitters.log_splitter
   :members: DateSplitter, RandomSplitter, ColdUsersSplitter
   :special-members:


