import keras
from keras import ops

from keras_hub.src.models.sam2.sam2_layers import SAM2Attention
from keras_hub.src.models.sam2.sam2_layers import SAM2TwoWayMultiHeadAttention


class TwoWayTransformer(keras.layers.Layer):
    """A two-way cross-attention transformer decoder.

    A transformer decoder that attends to an input image using
    queries whose positional embedding is supplied.

    The transformer decoder design is shown in [1]_.
    Each decoder layer performs 4 steps:
    (1) self-attention on the tokens,
    (2) cross-attention from tokens (as queries) to the image embedding,
    (3) a point-wise MLP updates each token, and
    (4) cross-attention from the image embedding (as queries) to tokens.

    This last step updates the image embedding with prompt information.
    Each self/cross-attention and MLP has a residual connection
    and layer normalization.

    Args:
        depth: int. The number of layers in the transformer.
        embedding_dim: int. The number of features of the input image
            and point embeddings.
        num_heads: int. Number of heads to use in the attention layers.
        mlp_dim: int. The number of units in the hidden layer of the MLP block.
        activation: str. The activation of the MLP block's output layer.
        attention_downsample_rate: int. The downsample rate
        of the attention layers.

    """

    def __init__(
        self,
        depth,
        embedding_dim,
        num_heads,
        mlp_dim,
        activation="relu",
        attention_downsample_rate=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.activation = activation
        self.attention_downsample_rate = attention_downsample_rate

        self.layers = []
        for i in range(depth):
            self.layers.append(
                SAM2TwoWayMultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embedding_dim // num_heads,  # Key dim per head
                    intermediate_dim=mlp_dim,
                    embedding_dim=embedding_dim,
                    skip_first_layer_pe=(i == 0),
                    attention_downsample_rate=attention_downsample_rate,
                    activation=activation,
                    dtype=self.dtype_policy,
                )
            )

        self.final_attn_token_to_image = SAM2Attention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            embedding_dim=embedding_dim,
            downsample_rate=attention_downsample_rate,
            dtype=self.dtype_policy,
        )

        self.norm_final_attn = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )

    def build(self, input_shape=None):
        for layer in self.layers:
            layer.build()
        self.final_attn_token_to_image.build()
        self.norm_final_attn.build((None, None, self.embedding_dim))
        self.built = True

    def call(self, image_embedding, image_pe, point_embedding):
        shape_img = ops.shape(image_embedding)
        batch_size, h, w, c = (
            shape_img[0],
            shape_img[1],
            shape_img[2],
            shape_img[3],
        )

        image_embedding = ops.reshape(image_embedding, (batch_size, h * w, c))
        image_pe = ops.reshape(image_pe, (batch_size, h * w, c))

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        q = queries + point_embedding
        k = keys + image_pe
        v = keys

        attn_out = self.final_attn_token_to_image(query=q, key=k, value=v)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "activation": self.activation,
                "attention_downsample_rate": self.attention_downsample_rate,
            }
        )
        return config
