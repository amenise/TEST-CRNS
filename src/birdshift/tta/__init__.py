"""Test-time adaptation methods with a common interface (see base.TTAMethod)."""

from .anchored import Anchored
from .base import TTAMethod
from .cotta import CoTTA
from .eata import EATA, RDumb
from .geo import GeoAnchored, GeoEATA, GeoTENT
from .lame import LAME
from .sar import SAR
from .simple import TENT, BNAdapt, PseudoLabel, Source
from .t3a import T3A

METHODS = {
    c.name: c
    for c in [
        Source,
        BNAdapt,
        TENT,
        PseudoLabel,
        EATA,
        SAR,
        T3A,
        LAME,
        CoTTA,
        RDumb,
        Anchored,
        GeoTENT,
        GeoEATA,
        GeoAnchored,
    ]
}


def build(method: str, source_model, **hp) -> TTAMethod:
    """`method` may carry an ablation suffix for Anchored TTA: anchored-noanchor, anchored-nofilter, anchored-noteacher."""
    base, _, variant = method.partition("-")
    if variant:
        hp = {**hp, {"noanchor": "use_anchor", "nofilter": "use_filter", "noteacher": "use_teacher"}[variant]: False}
    m = METHODS[base](source_model, **hp)
    m.name = method
    return m
