import torch
import torch.nn as nn
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding
from transformers import HfArgumentParser
from .utils.config import Configs


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    exp_setting=configs.exp_setting,
                )
                for l in range(configs.e_layer)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projection_head = nn.Linear(configs.d_model, configs.c_out)

    def forward(
        self, x_enc, x_mark_enc=None, enc_self_mask=None, return_features=False
    ):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        features, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection_head(features)
        if return_features:
            return enc_out, features
        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out  # [B, L, D]


def test():
    configs = Configs()
    model = Model(configs)
    x_enc = torch.randn(16, 12, 1)
    x_mark_enc = torch.randn(16, 12, 1)
    enc_self_mask = torch.ones(16, 12, 12).bool()
    out = model(x_enc, x_mark_enc, enc_self_mask)
    print(out.shape)  # 6,12,1


if __name__ == "__main__":
    test()
