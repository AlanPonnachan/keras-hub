import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.sam2.sam2_layers import SAM2MemoryAttentionLayer


@keras_hub_export("keras_hub.layers.SAM2MemoryAttention")
class SAM2MemoryAttention(keras.layers.Layer):
    """Memory Attention Module for SAM2.

    Stacks multiple SAM2MemoryAttentionLayers.

    Args:
        d_model: int.
        num_layers: int.
        num_heads: int.
        dim_feedforward: int.
        dropout: float.
        activation: str.
        pos_enc_at_input: bool. Whether to add PE to input before layers.
        pos_enc_at_attn: bool.
        pos_enc_at_cross_attn_keys: bool.
        pos_enc_at_cross_attn_queries: bool.
        rope_theta: float.
        rope_feat_sizes: tuple.
    """

    def __init__(
        self,
        d_model,
        num_layers,
        num_heads,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        pos_enc_at_input=True,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        rope_theta=10000.0,
        rope_feat_sizes=(64, 64),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.pos_enc_at_input = pos_enc_at_input
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.rope_theta = rope_theta
        self.rope_feat_sizes = rope_feat_sizes

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                SAM2MemoryAttentionLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    pos_enc_at_attn=pos_enc_at_attn,
                    pos_enc_at_cross_attn_keys=pos_enc_at_cross_attn_keys,
                    pos_enc_at_cross_attn_queries=pos_enc_at_cross_attn_queries,
                    rope_theta=rope_theta,
                    rope_feat_sizes=rope_feat_sizes,
                    dtype=self.dtype_policy,
                )
            )

        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )

    def build(self, input_shape=None):
        for layer in self.layers:
            layer.build()
        self.norm.build((None, None, self.d_model))
        self.built = True

    def call(
        self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0
    ):
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                num_k_exclude_rope=num_obj_ptr_tokens,
            )

        output = self.norm(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "activation": self.activation,
                "pos_enc_at_input": self.pos_enc_at_input,
                "pos_enc_at_attn": self.pos_enc_at_attn,
                "pos_enc_at_cross_attn_keys": self.pos_enc_at_cross_attn_keys,
                "pos_enc_at_cross_attn_queries": (
                    self.pos_enc_at_cross_attn_queries
                ),
                "rope_theta": self.rope_theta,
                "rope_feat_sizes": self.rope_feat_sizes,
            }
        )
        return config
