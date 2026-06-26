from .base import LUTModuleBase
from .pairwise import AbsDiffLUT, PAIRWISE_ANCHOR_POLICIES, PairwiseLUT, PairwiseRoute, PairwiseWalshLUT

PairwiseLinear = PairwiseLUT
PairwiseWalshLinear = PairwiseWalshLUT

__all__ = [
    "AbsDiffLUT",
    "LUTModuleBase",
    "PAIRWISE_ANCHOR_POLICIES",
    "PairwiseLinear",
    "PairwiseLUT",
    "PairwiseRoute",
    "PairwiseWalshLinear",
    "PairwiseWalshLUT",
]
