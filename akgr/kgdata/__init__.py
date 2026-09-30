print('# akgr/kgdata/__init__')
from .kgclass import GraphSampler
from .kgclass import KG


def load_kg(*args, **kwargs):
    """Lazily import PyKEEN-backed dataset loading.

    Serving only needs pre-built ``<dataset>.pkl`` objects, so importing the
    package must not eagerly import the optional PyKEEN dataset stack.
    """
    from .load_kg_util import load_kg as _load_kg
    return _load_kg(*args, **kwargs)