from dataclasses import dataclass


@dataclass
class TTTLayerCfg:
    model_dim: int
    num_heads: int
    mini_batch_size: int = 64
    ttt_base_lr: float = 0.1
    inter_multi: int = 1
    decouple_weight_norm: bool = True
    use_muon: bool = True
