Utils
=======


Time Smoothing
___________________

``time`` module provides function to apply time smoothing to item or interaction relevance.

.. autofunction:: replay.utils.time.smoothe_time

.. autofunction:: replay.utils.time.get_item_recency


Serializer
___________________

You can save trained models to disk and restore them later with ``save`` and ``load`` functions.

.. autofunction:: replay.utils.model_handler.save

.. autofunction:: replay.utils.model_handler.load


Distributions
___________________

Item Distribution
-------------------------------------------

Calculates item popularity in recommendations using 10 popularity bins.

.. autofunction:: replay.utils.distributions.item_distribution

You can plot the result. Here is the example for MovieLens log.

.. image:: /images/item_pop.jpg

.. autofunction:: replay.utils.distributions.plot_item_dist