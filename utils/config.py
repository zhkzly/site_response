from dataclasses import dataclass

@dataclass
class Configs:
    pred_len: int = 12
    output_attention: bool = False
    enc_in: int = 1
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 128
    e_layers: int = 2
    dropout: float = 0.1
    activation: str = 'gelu'
    factor: int = 5
    c_out: int = 1
    embed: str = 'timeF'
    freq: str = 'h'
    exp_setting: int = 0