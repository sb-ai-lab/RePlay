Distributions
=========================

Item Distribution
-------------------------------------------

Calculates item popularity in recommendations using 10 popularity bins.

.. autofunction:: replay.distributions.item_distribution

You can plot the result. Here is the example for MovieLens log.

.. image:: /images/item_pop.jpg

.. autofunction:: replay.distributions.plot_item_dist


User Distribution
-------------------------------------------------------------------

.. automethod:: replay.metrics.base_metric.Metric.user_distribution

If you plot this, you can get something like

.. image:: /images/user_dist.jpg

.. autofunction:: replay.distributions.plot_user_dist
