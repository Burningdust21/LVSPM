from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class LVSPM:
    color: Float[Tensor, "batch view 3 height width"]
    extrinsic: Float[Tensor, "batch view 4 4"] | None
    intrinsic: Float[Tensor, "batch view 3 3"] | None
    pose_encoding: Float[Tensor, "batch view 9"] | None
