Параметры сессии
==================

Spark
------

Библиотека использует класс ``session_handler.State`` для доступа модулей к единой спарк-сессии.
Для использования своей спарк сессии необходимо передать её в класс.

.. code-block:: python

    from replay.session_handler import State
    State(myspark)

.. autoclass:: replay.session_handler.State

Логирование
------------

Получить дефолтный логгер и изменить уровень логирования можно через имя ``replay``.
По умолчанию, он пишет на уровне ``logging.INFO``.

.. code-block:: python

    import logging
    logger = logging.getLogger("replay")
    logger.setLevel(logging.DEBUG)