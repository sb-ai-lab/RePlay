"""Neural building blocks for recommendation models.

The model subpackages require torch and lightning. The compiled-artifact
serving path (``replay.nn.compiled``: ``load``, ``encode_inputs``,
``ItemEmbeddings``) deliberately does not, so serving environments can
skip the training stack entirely.
"""
