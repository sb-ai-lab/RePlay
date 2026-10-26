from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from .bert4rec import BERT4Rec
    from .sasrec import SasRec
