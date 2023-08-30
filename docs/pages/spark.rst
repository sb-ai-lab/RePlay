Settings
==================

Spark session
--------------

This library uses ``session_handler.State`` to provide universal access to the same session for all modules.
Default session will be created automatically and can be accessed as a ``session`` attribute.

.. code-block:: python

    from replay.utils.session_handler import State
    State().session

There is also a helper function to provide basic settings for the creation of Spark session

.. autofunction:: replay.utils.session_handler.get_spark_session

You can pass any Spark session to ``State`` for it to be available in library.

.. code-block:: python

    from replay.utils.session_handler import get_spark_session
    session = get_spark_session(2)
    State(session)


.. autoclass:: replay.utils.session_handler.State

Logging
------------

Logger name is ``replay``.
Default level is ``logging.INFO``.

.. code-block:: python

    import logging
    logger = logging.getLogger("replay")
    logger.setLevel(logging.DEBUG)