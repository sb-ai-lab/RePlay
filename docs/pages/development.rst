Development
============

For development details please check our
`contributing guidelines <https://github.com/sb-ai-lab/RePlay/blob/main/CONTRIBUTING.md>`_.

Adding new model
-------------------

Usually for new model we need updates in ``fit``, ``predict`` and ``predict_pairs``.

Typical function chains for ``fit`` and ``predict[_pairs]``::

    fit->_fit_wrap->_fit
    predict[_pairs]->_predict[_pairs]_wrap->_predict[_pairs]

For *LightFMWrap*::

    predict[_pairs]->_predict[_pairs]_wrap->_predict[_pairs]->_predict_selected_pairs

For *Word2VecRec* and models inherited from *NeighbourRec*::

    predict[_pairs]->_predict[_pairs]_wrap->_predict[_pairs]->_predict_pairs_inner

For models inherited from *BaseTorchRec*::

    fit->_fit_wrap->_fit->train
    predict[_pairs]->_predict[_pairs]_wrap->_predict[_pairs]->_predict_by_user[_pairs]->_predict_pairs_inner

Currently *AssociationRulesItemRec* doesn't have ``predict[_pairs]`` interface.
*ClusterRec*, *ImplicitWrap* and *RandomRec* currently doesn't have ``predict_pairs`` interface.

Current model inheritance in RePlay:

.. image:: /images/model_inheritance.jpg

`Current framework architecture <https://miro.com/app/board/uXjVOhTSHK0=/?share_link_id=748466292621>`_
