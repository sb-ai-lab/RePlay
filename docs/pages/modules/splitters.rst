splitters
==========

.. automodule:: sponge_bob_magic.splitters

Все модули возвращают сплиты по методу ``split``.

.. autofunction:: sponge_bob_magic.splitters.base_splitter.Splitter.split

Базовые сплиттеры
------------------

.. automodule:: sponge_bob_magic.splitters.log_splitter
   :members: DateSplitter, RandomSplitter, ColdUsersSplitter
   :special-members:

Юзер-сплиттеры
-----------------

.. automodule:: sponge_bob_magic.splitters.user_log_splitter
   :members: RandomUserSplitter, TimeUserSplitter

---------------

Все сплиттеры этого вида унаследованы от ``UserSplitter`` и инициализируются одинаково:

.. autoclass:: sponge_bob_magic.splitters.user_log_splitter.UserSplitter
   :special-members: __init__