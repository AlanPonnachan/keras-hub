import math

import keras
from keras import ops
from keras import random


class MLP(keras.layers.Layer):
    """A MLP block for SAM2.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        output_dim: int. The number of units in the output layer.
        num_layers: int. The total number of dense layers to use.
        activation: str. Activation to use in the hidden layers.
            Default is `"relu"`.
        sigmoid_output: bool. Whether to apply sigmoid activation to the output.
            Default is `False`.
    """

    def __init__(
        self,
        hidden_dim,
        output_dim,
        num_layers,
        activation="relu",
        sigmoid_output=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.sigmoid_output = sigmoid_output

        self.layers = []
        for _ in range(num_layers - 1):
            self.layers.append(
                keras.layers.Dense(hidden_dim, dtype=self.dtype_policy)
            )
            self.layers.append(
                keras.layers.Activation(activation, dtype=self.dtype_policy)
            )

        self.layers.append(
            keras.layers.Dense(output_dim, dtype=self.dtype_policy)
        )

        if sigmoid_output:
            self.layers.append(
                keras.layers.Activation("sigmoid", dtype=self.dtype_policy)
            )

        self.mlp_block = keras.models.Sequential(self.layers)
        self.mlp_block.dtype_policy = self.dtype_policy

    def build(self, input_shape):
        self.mlp_block.build(input_shape)
        self.built = True

    def call(self, x):
        return self.mlp_block(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "sigmoid_output": self.sigmoid_output,
            }
        )
        return config


class FixedFrequencyPositionalEmbeddings(keras.layers.Layer):
    """Fixed sinusoidal positional encoding for 2D images.

    Corresponds to `PositionEmbeddingSine` in SAM2.

    Args:
        num_positional_features: int. Number of positional features
        in the output. Must be even. The actual output depth will be
        `num_positional_features`.
        temperature: int. Temperature for the sine frequency. Default 10000.
        normalize: bool. Whether to normalize coordinates. Default True.
        scale: float. Scale factor if normalize is True. Default 2 * pi.
    """

    def __init__(
        self,
        num_positional_features,
        temperature=10000,
        normalize=True,
        scale=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_positional_features = num_positional_features
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def call(self, x):
        compute_dtype = "float32"

        shape = ops.shape(x)
        height, width = shape[1], shape[2]

        num_dim = self.num_positional_features // 2

        mask = ops.ones((1, height, width), dtype=compute_dtype)
        y_embed = ops.cumsum(mask, axis=1) - 0.5
        x_embed = ops.cumsum(mask, axis=2) - 0.5

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.arange(num_dim, dtype=compute_dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_dim)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x_sin = ops.sin(pos_x[..., 0::2])
        pos_x_cos = ops.cos(pos_x[..., 1::2])

        pos_x = ops.reshape(
            ops.stack([pos_x_sin, pos_x_cos], axis=-1), (1, height, width, -1)
        )

        pos_y_sin = ops.sin(pos_y[..., 0::2])
        pos_y_cos = ops.cos(pos_y[..., 1::2])
        pos_y = ops.reshape(
            ops.stack([pos_y_sin, pos_y_cos], axis=-1), (1, height, width, -1)
        )

        pos = ops.concatenate([pos_y, pos_x], axis=-1)

        batch_size = shape[0]
        pos = ops.broadcast_to(
            pos, (batch_size, height, width, self.num_positional_features)
        )

        return ops.cast(pos, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_positional_features": self.num_positional_features,
                "temperature": self.temperature,
                "normalize": self.normalize,
                "scale": self.scale,
            }
        )
        return config


class RandomFrequencyPositionalEmbeddings(keras.layers.Layer):
    """Positional encoding using random spatial frequencies.

    Args:
        num_positional_features: int. Number of positional features
            in the output.
        scale: float. The standard deviation of the random frequencies.
    """

    def __init__(self, num_positional_features, scale, **kwargs):
        super().__init__(**kwargs)
        self.num_positional_features = num_positional_features
        self.scale = scale
        self.positional_encoding_gaussian_matrix = self.add_weight(
            name="positional_encoding_gaussian_matrix",
            shape=(2, self.num_positional_features),
            dtype=self.variable_dtype,
            trainable=False,
            initializer=keras.initializers.get("normal"),
        )
        self.positional_encoding_gaussian_matrix.assign(
            self.positional_encoding_gaussian_matrix * self.scale
        )

    def build(self, input_shape=None):
        self.built = True

    def _positional_encodings(self, coords):
        coords = 2 * coords - 1
        coords = coords @ ops.cast(
            self.positional_encoding_gaussian_matrix, dtype=self.compute_dtype
        )
        coords = 2 * math.pi * coords
        return ops.concatenate([ops.sin(coords), ops.cos(coords)], axis=-1)

    def call(self, size):
        return self.encode_image(size)

    def encode_image(self, size):
        height, width = size
        compute_dtype = "float32"

        grid = ops.ones(shape=(height, width), dtype=compute_dtype)
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / ops.cast(height, compute_dtype)
        x_embed = x_embed / ops.cast(width, compute_dtype)

        encoding = self._positional_encodings(
            ops.cast(ops.stack([x_embed, y_embed], axis=-1), self.compute_dtype)
        )
        return encoding

    def encode_coordinates(self, coords_input, image_size):
        coords_normalized = ops.stack(
            [
                coords_input[..., 0] / image_size[1],
                coords_input[..., 1] / image_size[0],
            ],
            axis=-1,
        )
        return self._positional_encodings(coords_normalized)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_positional_features": self.num_positional_features,
                "scale": self.scale,
            }
        )
        return config


class DropPath(keras.layers.Layer):
    """A stochastic depth layer.

    This layer randomly drops input paths (samples) during training with a
    specified probability `rate`. It is commonly used in deep residual
    networks to improve generalization.

    Args:
        rate: float. The probability of dropping an input path.

    """

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, x, training=None):
        if not training or self.rate == 0.0:
            return x
        keep_prob = 1 - self.rate
        shape = (ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = random.bernoulli(
            shape, probability=keep_prob, dtype=self.compute_dtype
        )
        x = x / keep_prob
        x = x * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


def compute_axial_rope_cache(
    dim, height, width, theta=10000.0, compute_dtype="float32"
):
    dim_t = dim // 2
    theta_dim = dim_t // 2
    freqs = 1.0 / (
        theta ** (ops.arange(0, theta_dim, dtype=compute_dtype) * 2 / theta_dim)
    )

    t_x = ops.arange(width, dtype=compute_dtype)
    t_y = ops.arange(height, dtype=compute_dtype)

    t_x = ops.tile(t_x[None, :], (height, 1))
    t_y = ops.tile(t_y[:, None], (1, width))

    t_x = ops.reshape(t_x, (-1,))
    t_y = ops.reshape(t_y, (-1,))

    theta_x = ops.outer(t_x, freqs)
    theta_y = ops.outer(t_y, freqs)

    theta_all = ops.concatenate([theta_x, theta_y], axis=-1)

    cos = ops.cos(theta_all)
    sin = ops.sin(theta_all)

    cos = ops.repeat(cos, 2, axis=-1)
    sin = ops.repeat(sin, 2, axis=-1)

    return cos, sin


class SAM2Attention(keras.layers.Layer):
    """Multi-Head Attention with downsampling.

    Args:
        num_heads: int. Number of attention heads.
        key_dim: int. Size of each attention head for query, key, and value.
        downsample_rate: int. The factor by which to downscale input features.
        embedding_dim: int. The output embedding dimension.
        kv_in_dim: int, optional. Dimension of key/value inputs.
        dropout: float. Dropout probability.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        embedding_dim,
        downsample_rate=1,
        kv_in_dim=None,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.dropout = dropout

        self.internal_dim = embedding_dim // downsample_rate

        self.q_proj = keras.layers.Dense(
            self.internal_dim, dtype=self.dtype_policy
        )
        self.k_proj = keras.layers.Dense(
            self.internal_dim, dtype=self.dtype_policy
        )
        self.v_proj = keras.layers.Dense(
            self.internal_dim, dtype=self.dtype_policy
        )
        self.out_proj = keras.layers.Dense(
            embedding_dim, dtype=self.dtype_policy
        )

        if self.dropout > 0:
            self.dropout_layer = keras.layers.Dropout(
                dropout, dtype=self.dtype_policy
            )

    def build(self, input_shape=None, **kwargs):
        self.q_proj.build((None, None, self.embedding_dim))
        self.k_proj.build((None, None, self.kv_in_dim))
        self.v_proj.build((None, None, self.kv_in_dim))
        self.out_proj.build((None, None, self.internal_dim))
        self.built = True

    def _separate_heads(self, x):
        shape = ops.shape(x)
        batch_size, n, channels = shape[0], shape[1], shape[2]
        x = ops.reshape(
            x, (batch_size, n, self.num_heads, channels // self.num_heads)
        )
        return ops.transpose(x, axes=(0, 2, 1, 3))

    def _recombine_heads(self, x):
        shape = ops.shape(x)
        batch_size, num_heads, n, key_dim = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )
        x = ops.transpose(x, axes=(0, 2, 1, 3))
        return ops.reshape(x, (batch_size, n, num_heads * key_dim))

    def call(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._separate_heads(q)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        scale = ops.cast(ops.shape(q)[-1], self.compute_dtype)
        q = q / ops.sqrt(scale)

        attn_score = ops.matmul(q, ops.transpose(k, axes=(0, 1, 3, 2)))
        attn_weights = ops.softmax(attn_score, axis=-1)

        if self.dropout > 0:
            attn_weights = self.dropout_layer(attn_weights)

        out = ops.matmul(attn_weights, v)
        out = self._recombine_heads(out)
        return self.out_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "embedding_dim": self.embedding_dim,
                "downsample_rate": self.downsample_rate,
                "kv_in_dim": self.kv_in_dim,
                "dropout": self.dropout,
            }
        )
        return config


class RoPEAttention(SAM2Attention):
    """Attention layer with Rotary Positional Encoding (RoPE).

    This layer extends `SAM2Attention` by applying rotary positional
    embeddings to the query and key tensors before computing attention
    scores. It is used in the memory attention mechanism of SAM2.

    Args:
        rope_theta: float. The base frequency parameter for RoPE.
        rope_k_repeat: bool. Whether to repeat the RoPE frequencies for keys
            to match the sequence length of queries.
        feat_sizes: tuple. The spatial dimensions (height, width) of the
            feature map used to generate RoPE frequencies.

    """

    def __init__(
        self,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(64, 64),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rope_theta = rope_theta
        self.rope_k_repeat = rope_k_repeat
        self.feat_sizes = feat_sizes

    def build(self, input_shape=None, **kwargs):
        super().build(input_shape, **kwargs)
        head_dim = self.internal_dim // self.num_heads
        self.cos_cache, self.sin_cache = compute_axial_rope_cache(
            head_dim,
            self.feat_sizes[0],
            self.feat_sizes[1],
            self.rope_theta,
            compute_dtype=self.compute_dtype,
        )

    def _apply_rope(self, x, cos, sin):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotate_x = ops.stack([-x2, x1], axis=-1)
        rotate_x = ops.reshape(rotate_x, ops.shape(x))

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return x * cos + rotate_x * sin

    def call(self, query, key, value, num_k_exclude_rope=0):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._separate_heads(q)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        cos = ops.cast(self.cos_cache, self.compute_dtype)
        sin = ops.cast(self.sin_cache, self.compute_dtype)

        q = self._apply_rope(q, cos, sin)

        k_len = ops.shape(k)[2]
        num_k_rope = k_len - num_k_exclude_rope
        k_spatial = k[..., :num_k_rope, :]
        k_other = k[..., num_k_rope:, :]

        if self.rope_k_repeat:
            repeats = num_k_rope // ops.shape(cos)[0]
            cos_k = ops.tile(cos, (repeats, 1))
            sin_k = ops.tile(sin, (repeats, 1))
        else:
            cos_k = cos
            sin_k = sin

        k_spatial = self._apply_rope(k_spatial, cos_k, sin_k)
        k = ops.concatenate([k_spatial, k_other], axis=2)

        scale = ops.cast(ops.shape(q)[-1], self.compute_dtype)
        q = q / ops.sqrt(scale)

        attn_score = ops.matmul(q, ops.transpose(k, axes=(0, 1, 3, 2)))
        attn_weights = ops.softmax(attn_score, axis=-1)

        if self.dropout > 0:
            attn_weights = self.dropout_layer(attn_weights)

        out = ops.matmul(attn_weights, v)
        out = self._recombine_heads(out)
        return self.out_proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rope_theta": self.rope_theta,
                "rope_k_repeat": self.rope_k_repeat,
                "feat_sizes": self.feat_sizes,
            }
        )
        return config


class SAM2TwoWayMultiHeadAttention(keras.layers.Layer):
    """Two-way multi-head attention layer for SAM2.

    This layer implements a block that performs self-attention on sparse
    prompts, cross-attention between prompts and image embeddings, and
    MLP updates. It allows bidirectional information flow between prompts
    and image features.

    Args:
        num_heads: int. The number of attention heads.
        key_dim: int. The size of each attention head for query, key, and
            value.
        intermediate_dim: int. The number of units in the hidden layer of
            the MLP block.
        embedding_dim: int. The number of input/output features.
        skip_first_layer_pe: bool. Whether to skip adding positional
            embeddings in the first self-attention layer.
        attention_downsample_rate: int. The factor by which to downsample
            input features in attention layers.
        activation: str. The activation function to use in the MLP block.

    """

    def __init__(
        self,
        num_heads,
        key_dim,
        intermediate_dim,
        embedding_dim,
        skip_first_layer_pe=False,
        attention_downsample_rate=2,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.intermediate_dim = intermediate_dim
        self.embedding_dim = embedding_dim
        self.skip_first_layer_pe = skip_first_layer_pe
        self.attention_downsample_rate = attention_downsample_rate
        self.activation = activation

        self.self_attention = SAM2Attention(
            num_heads=num_heads,
            key_dim=key_dim,
            embedding_dim=embedding_dim,
            downsample_rate=1,
            dtype=self.dtype_policy,
        )
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )

        self.cross_attention_token_to_image = SAM2Attention(
            num_heads=num_heads,
            key_dim=key_dim,
            embedding_dim=embedding_dim,
            downsample_rate=attention_downsample_rate,
            dtype=self.dtype_policy,
        )
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )

        self.mlp_block = MLP(
            hidden_dim=intermediate_dim,
            output_dim=embedding_dim,
            num_layers=2,
            activation=activation,
            dtype=self.dtype_policy,
        )
        self.layer_norm3 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )

        self.cross_attention_image_to_token = SAM2Attention(
            num_heads=num_heads,
            key_dim=key_dim,
            embedding_dim=embedding_dim,
            downsample_rate=attention_downsample_rate,
            dtype=self.dtype_policy,
        )
        self.layer_norm4 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )

    def build(self, input_shape=None):
        self.self_attention.build()
        self.layer_norm1.build((None, None, self.embedding_dim))
        self.cross_attention_token_to_image.build()
        self.layer_norm2.build((None, None, self.embedding_dim))
        self.mlp_block.build((None, None, self.embedding_dim))
        self.layer_norm3.build((None, None, self.embedding_dim))
        self.cross_attention_image_to_token.build()
        self.layer_norm4.build((None, None, self.embedding_dim))
        self.built = True

    def call(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            attn_out = self.self_attention(
                query=queries, key=queries, value=queries
            )
        else:
            q = queries + query_pe
            attn_out = self.self_attention(query=q, key=q, value=queries)
        queries = queries + attn_out
        queries = self.layer_norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attention_token_to_image(
            query=q, key=k, value=keys
        )
        queries = queries + attn_out
        queries = self.layer_norm2(queries)

        mlp_out = self.mlp_block(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attention_image_to_token(
            query=k, key=q, value=queries
        )
        keys = keys + attn_out
        keys = self.layer_norm4(keys)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "intermediate_dim": self.intermediate_dim,
                "embedding_dim": self.embedding_dim,
                "skip_first_layer_pe": self.skip_first_layer_pe,
                "attention_downsample_rate": self.attention_downsample_rate,
                "activation": self.activation,
            }
        )
        return config


class HieraPatchEmbed(keras.layers.Layer):
    """Image to Patch Embedding for Hiera.

    Args:
        kernel_size: tuple. Kernel size of the projection layer.
        stride: tuple. Stride of the projection layer.
        padding: str. Padding strategy ("same" or "valid").
        embed_dim: int. Patch embedding dimension.
    """

    def __init__(
        self,
        kernel_size=(7, 7),
        stride=(4, 4),
        padding="same",
        embed_dim=768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.embed_dim = embed_dim

        self.proj = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape=None):
        self.proj.build(input_shape)
        self.built = True

    def call(self, x):
        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "embed_dim": self.embed_dim,
            }
        )
        return config


def window_partition(x, window_size):
    """Partitions tensor into windows."""
    input_shape = ops.shape(x)
    B, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

    Hp = H + pad_h
    Wp = W + pad_w

    x = ops.reshape(
        x,
        (B, Hp // window_size, window_size, Wp // window_size, window_size, C),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = ops.reshape(x, (-1, window_size, window_size, C))

    return windows, Hp, Wp


def window_unpartition(windows, window_size, Hp, Wp, H, W):
    """Reverses window partition."""
    input_shape = ops.shape(windows)
    C = input_shape[3]

    x = ops.reshape(
        windows,
        (-1, Hp // window_size, Wp // window_size, window_size, window_size, C),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (-1, Hp, Wp, C))

    x = x[:, :H, :W, :]

    return x


class HieraMultiScaleAttention(keras.layers.Layer):
    """Multi-scale (windowed) attention layer for the Hiera backbone.

    This layer performs multi-head attention, optionally within local
    windows and with query pooling (downsampling).

    Args:
        dim: int. The input feature dimension.
        dim_out: int. The output feature dimension.
        num_heads: int. The number of attention heads.
        q_stride: tuple or None. The stride for pooling the query, used for
            downsampling. If None, no pooling is applied.
        window_size: int. The size of the window for local attention. If 0,
            global attention is used.

    """

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        q_stride=None,
        window_size=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_stride = q_stride
        self.window_size = window_size

        self.qkv = keras.layers.Dense(dim_out * 3, dtype=self.dtype_policy)
        self.proj = keras.layers.Dense(dim_out, dtype=self.dtype_policy)

        if self.q_stride:
            self.pool = keras.layers.MaxPooling2D(
                pool_size=q_stride,
                strides=q_stride,
                padding="valid",
                dtype=self.dtype_policy,
            )

    def build(self, input_shape=None):
        self.qkv.build((None, None, None, self.dim))
        self.proj.build((None, None, None, self.dim_out))
        if self.q_stride:
            self.pool.build((None, None, None, self.dim_out))
        self.built = True

    def call(self, x):
        B, H, W, _ = (
            ops.shape(x)[0],
            ops.shape(x)[1],
            ops.shape(x)[2],
            ops.shape(x)[3],
        )

        qkv = self.qkv(x)
        head_dim = self.dim_out // self.num_heads
        qkv = ops.reshape(qkv, (B, H * W, 3, self.num_heads, head_dim))

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        if self.q_stride:
            q = ops.reshape(q, (B, H, W, self.dim_out))
            q = self.pool(q)
            H_new, W_new = ops.shape(q)[1], ops.shape(q)[2]
            q = ops.reshape(q, (B, H_new * W_new, self.num_heads, head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        scale = ops.cast(head_dim, self.compute_dtype)
        q = q / ops.sqrt(scale)
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        attn = ops.softmax(attn, axis=-1)
        x = ops.matmul(attn, v)

        x = ops.transpose(x, (0, 2, 1, 3))

        if self.q_stride:
            H_out = H // self.q_stride[0]
            W_out = W // self.q_stride[1]
        else:
            H_out, W_out = H, W

        x = ops.reshape(x, (B, H_out, W_out, self.dim_out))

        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_out": self.dim_out,
                "num_heads": self.num_heads,
                "q_stride": self.q_stride,
                "window_size": self.window_size,
            }
        )
        return config


class HieraBlock(keras.layers.Layer):
    """Transformer block for the Hiera backbone.

    This block consists of a multi-scale attention layer (optionally
    windowed) and an MLP block, with residual connections and layer
    normalization. It supports stochastic depth (DropPath).

    Args:
        dim: int. The input feature dimension.
        dim_out: int. The output feature dimension.
        num_heads: int. The number of attention heads.
        mlp_ratio: float. The ratio of the hidden dimension in the MLP to
            the input dimension.
        drop_path: float. The probability of dropping paths in the DropPath
            layer.
        q_stride: tuple or None. The stride for pooling the query in the
            attention layer.
        window_size: int. The window size for local attention.
        activation: str. The activation function to use in the MLP.

    """

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        q_stride=None,
        window_size=0,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        self.q_stride = q_stride
        self.window_size = window_size
        self.activation = activation

        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6, dtype=self.dtype_policy
        )
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6, dtype=self.dtype_policy
        )

        self.attn = HieraMultiScaleAttention(
            dim=dim,
            dim_out=dim_out,
            num_heads=num_heads,
            q_stride=q_stride,
            window_size=window_size,
            dtype=self.dtype_policy,
        )

        self.drop_path_layer = DropPath(drop_path, dtype=self.dtype_policy)

        self.mlp = MLP(
            hidden_dim=int(dim_out * mlp_ratio),
            output_dim=dim_out,
            num_layers=2,
            activation=activation,
            dtype=self.dtype_policy,
        )

        if dim != dim_out:
            self.proj = keras.layers.Dense(dim_out, dtype=self.dtype_policy)
        else:
            self.proj = None

        if self.q_stride:
            self.pool = keras.layers.MaxPooling2D(
                pool_size=q_stride,
                strides=q_stride,
                padding="valid",
                dtype=self.dtype_policy,
            )
        else:
            self.pool = None

    def build(self, input_shape=None):
        self.norm1.build(input_shape)
        self.attn.build(input_shape)
        if self.dim != self.dim_out:
            self.proj.build(input_shape)
        self.norm2.build((None, None, None, self.dim_out))
        self.mlp.build((None, None, None, self.dim_out))
        self.built = True

    def call(self, x):
        shortcut = x
        if self.dim != self.dim_out:
            shortcut = self.proj(shortcut)
        if self.pool:
            shortcut = self.pool(shortcut)

        x = self.norm1(x)

        if self.window_size > 0:
            windows, Hp, Wp = window_partition(x, self.window_size)
            x = self.attn(windows)

            if self.q_stride:
                ws_new = self.window_size // self.q_stride[0]
                shape_sc = ops.shape(shortcut)
                H_out, W_out = shape_sc[1], shape_sc[2]

                pad_h = (ws_new - H_out % ws_new) % ws_new
                pad_w = (ws_new - W_out % ws_new) % ws_new
                Hp_new = H_out + pad_h
                Wp_new = W_out + pad_w

                x = window_unpartition(x, ws_new, Hp_new, Wp_new, H_out, W_out)
            else:
                shape_x_orig = ops.shape(shortcut)
                H, W = shape_x_orig[1], shape_x_orig[2]
                x = window_unpartition(x, self.window_size, Hp, Wp, H, W)
        else:
            x = self.attn(x)

        x = shortcut + self.drop_path_layer(x)
        x = x + self.drop_path_layer(self.mlp(self.norm2(x)))

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "dim_out": self.dim_out,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "drop_path": self.drop_path,
                "q_stride": self.q_stride,
                "window_size": self.window_size,
                "activation": self.activation,
            }
        )
        return config


class FpnNeck(keras.layers.Layer):
    """Feature Pyramid Network (FPN) Neck for SAM2.

    This layer processes multi-scale feature maps from the backbone, fusing
    them in a top-down manner and adding positional encodings.

    Args:
        d_model: int. The output feature dimension for all levels.
        backbone_channel_list: list of ints. The channel dimensions of the
            input feature maps from the backbone stages.
        fpn_top_down_levels: list of ints or None. The indices of the
            levels to apply top-down fusion. If None, all levels are used.

    """

    def __init__(
        self,
        d_model,
        backbone_channel_list,
        fpn_top_down_levels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.backbone_channel_list = backbone_channel_list
        if fpn_top_down_levels is None:
            self.fpn_top_down_levels = list(range(len(backbone_channel_list)))
        else:
            self.fpn_top_down_levels = fpn_top_down_levels

        self.convs = []
        for _ in backbone_channel_list:
            self.convs.append(
                keras.layers.Conv2D(
                    filters=d_model,
                    kernel_size=1,
                    padding="valid",
                    dtype=self.dtype_policy,
                )
            )

        self.position_encoding = FixedFrequencyPositionalEmbeddings(
            num_positional_features=d_model,
            normalize=True,
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        for i, conv in enumerate(self.convs):
            conv.build(input_shape[i])
        self.position_encoding.build(None)
        self.built = True

    def call(self, xs):
        outputs = [None] * len(self.convs)
        pos_encs = [None] * len(self.convs)

        prev_features = None
        n = len(self.convs) - 1

        for i in range(n, -1, -1):
            x = xs[i]
            lateral = self.convs[i](x)

            if i in self.fpn_top_down_levels and prev_features is not None:
                target_h = ops.shape(lateral)[1]
                target_w = ops.shape(lateral)[2]

                top_down = ops.image.resize(
                    prev_features, (target_h, target_w), interpolation="nearest"
                )
                prev_features = lateral + top_down
            else:
                prev_features = lateral

            outputs[i] = prev_features
            pos_encs[i] = self.position_encoding(prev_features)

        return outputs, pos_encs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "backbone_channel_list": self.backbone_channel_list,
                "fpn_top_down_levels": self.fpn_top_down_levels,
            }
        )
        return config


class CXBlock(keras.layers.Layer):
    """ConvNeXt-style block for the SAM2 Memory Encoder.

    This block implements a residual block with a depthwise convolution,
    layer normalization, and two pointwise convolutions (MLP), similar to
    the ConvNeXt architecture.

    Args:
        dim: int. The number of input channels.
        kernel_size: int. The kernel size for the depthwise convolution.
        padding: str. The padding strategy for the depthwise convolution.
        drop_path: float. The probability of dropping paths in the DropPath
            layer.
        layer_scale_init_value: float. The initial value for layer scaling.
        use_dwconv: bool. Whether to use depthwise convolution.

    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding="same",
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.drop_path = drop_path
        self.layer_scale_init_value = layer_scale_init_value
        self.use_dwconv = use_dwconv

        if use_dwconv:
            self.dwconv = keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                padding=padding,
                dtype=self.dtype_policy,
            )
        else:
            self.dwconv = keras.layers.Conv2D(
                filters=dim,
                kernel_size=kernel_size,
                padding=padding,
                dtype=self.dtype_policy,
            )

        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-6, dtype=self.dtype_policy
        )
        self.pwconv1 = keras.layers.Dense(4 * dim, dtype=self.dtype_policy)
        self.act = keras.layers.Activation("gelu", dtype=self.dtype_policy)
        self.pwconv2 = keras.layers.Dense(dim, dtype=self.dtype_policy)

        if layer_scale_init_value > 0:
            self.gamma = self.add_weight(
                name="gamma",
                shape=(dim,),
                initializer=keras.initializers.Constant(layer_scale_init_value),
                trainable=True,
                dtype=self.variable_dtype,
            )
        else:
            self.gamma = None

        self.drop_path_layer = DropPath(drop_path, dtype=self.dtype_policy)

    def build(self, input_shape):
        self.dwconv.build(input_shape)
        self.norm.build(input_shape)
        self.pwconv1.build(input_shape)

        hidden_shape = list(input_shape)
        if hidden_shape[-1] is not None:
            hidden_shape[-1] = 4 * self.dim
        self.pwconv2.build(tuple(hidden_shape))

        self.built = True

    def call(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = x * self.gamma

        x = input + self.drop_path_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "drop_path": self.drop_path,
                "layer_scale_init_value": self.layer_scale_init_value,
                "use_dwconv": self.use_dwconv,
            }
        )
        return config


class MaskDownSampler(keras.layers.Layer):
    """Mask downsampler for the SAM2 Memory Encoder.

    This layer progressively downsamples a mask input using a series of
    convolutions, layer normalizations, and activations.

    Args:
        embed_dim: int. The number of output channels.
        kernel_size: int. The kernel size for the convolutional layers.
        stride: int. The stride for the convolutional layers.
        padding: str. The padding strategy for the convolutional layers.
        total_stride: int. The total downsampling factor.
        activation: str. The activation function to use.

    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding="same",
        total_stride=16,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.total_stride = total_stride
        self.activation = activation

        num_layers = int(math.log2(total_stride) // math.log2(stride))

        self.layers = []
        mask_in_chans = 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.layers.append(
                keras.layers.Conv2D(
                    filters=mask_out_chans,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    dtype=self.dtype_policy,
                )
            )
            self.layers.append(
                keras.layers.LayerNormalization(dtype=self.dtype_policy)
            )
            self.layers.append(
                keras.layers.Activation(activation, dtype=self.dtype_policy)
            )
            mask_in_chans = mask_out_chans

        self.layers.append(
            keras.layers.Conv2D(
                filters=embed_dim, kernel_size=1, dtype=self.dtype_policy
            )
        )
        self.encoder = keras.Sequential(self.layers)
        self.encoder.dtype_policy = self.dtype_policy

    def build(self, input_shape):
        self.encoder.build(input_shape)
        self.built = True

    def call(self, x):
        return self.encoder(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "total_stride": self.total_stride,
                "activation": self.activation,
            }
        )
        return config


class Hiera(keras.layers.Layer):
    """ "Hiera backbone for SAM2.

    This layer implements the hierarchical vision transformer (Hiera) used
    as the image encoder backbone in SAM2. It consists of a patch
    embedding layer followed by multiple stages of Hiera blocks.

    Args:
        embed_dim: int. The initial embedding dimension.
        num_heads: int. The initial number of attention heads.
        drop_path_rate: float. The stochastic depth rate.
        q_pool: int. The number of stages with query pooling.
        q_stride: tuple. The stride for query pooling.
        stages: tuple. The number of blocks in each stage.
        dim_mul: float. The multiplier for the embedding dimension at each
            stage transition.
        head_mul: float. The multiplier for the number of heads at each
            stage transition.
        window_pos_embed_bkg_spatial_size: tuple. The spatial size of the
            background positional embedding.
        window_spec: tuple. The window sizes for each stage.
        global_att_blocks: tuple. The indices of blocks using global
            attention.
        return_interm_layers: bool. Whether to return feature maps from
            all stages.

    """

    def __init__(
        self,
        embed_dim=96,
        num_heads=1,
        drop_path_rate=0.0,
        q_pool=3,
        q_stride=(2, 2),
        stages=(2, 3, 16, 3),
        dim_mul=2.0,
        head_mul=2.0,
        window_pos_embed_bkg_spatial_size=(14, 14),
        window_spec=(8, 4, 14, 7),
        global_att_blocks=(12, 16, 20),
        return_interm_layers=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.q_pool = q_pool
        self.q_stride = q_stride
        self.stages = stages
        self.dim_mul = dim_mul
        self.head_mul = head_mul
        self.window_pos_embed_bkg_spatial_size = (
            window_pos_embed_bkg_spatial_size
        )
        self.window_spec = window_spec
        self.global_att_blocks = global_att_blocks
        self.return_interm_layers = return_interm_layers

        depth = sum(stages)
        self.stage_ends = [
            sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)
        ]
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]

        self.patch_embed = HieraPatchEmbed(
            embed_dim=embed_dim, dtype=self.dtype_policy
        )

        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(
                1,
                window_pos_embed_bkg_spatial_size[0],
                window_pos_embed_bkg_spatial_size[1],
                embed_dim,
            ),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
        )
        self.pos_embed_window = self.add_weight(
            name="pos_embed_window",
            shape=(1, window_spec[0], window_spec[0], embed_dim),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
        )

        step = drop_path_rate / (depth - 1) if depth > 1 else 0

        self.blocks = []
        cur_stage = 1
        curr_embed_dim = embed_dim
        curr_num_heads = num_heads

        for i in range(depth):
            dim_out = curr_embed_dim
            window_size = window_spec[cur_stage - 1]

            if global_att_blocks is not None:
                if i in global_att_blocks:
                    window_size = 0

            if i - 1 in self.stage_ends:
                dim_out = int(curr_embed_dim * dim_mul)
                curr_num_heads = int(curr_num_heads * head_mul)
                cur_stage += 1

            block_dpr = i * step

            block = HieraBlock(
                dim=curr_embed_dim,
                dim_out=dim_out,
                num_heads=curr_num_heads,
                drop_path=block_dpr,
                q_stride=q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                dtype=self.dtype_policy,
            )

            self.blocks.append(block)
            curr_embed_dim = dim_out

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def build(self, input_shape):
        self.patch_embed.build(input_shape)
        current_dim = self.embed_dim
        current_shape = (None, None, None, current_dim)

        for block in self.blocks:
            block.build(current_shape)
            current_shape = (None, None, None, block.dim_out)

        self.built = True

    def _get_pos_embed(self, h, w):
        pos_embed = ops.image.resize(
            self.pos_embed, (h, w), interpolation="bicubic"
        )

        ws_h = self.pos_embed_window.shape[1]
        ws_w = self.pos_embed_window.shape[2]

        tile_h = h // ws_h
        tile_w = w // ws_w

        window_embed_tiled = ops.tile(
            self.pos_embed_window, (1, tile_h, tile_w, 1)
        )

        return pos_embed + window_embed_tiled

    def call(self, x):
        x = self.patch_embed(x)

        shape = ops.shape(x)
        h, w = shape[1], shape[2]

        pos_embed = self._get_pos_embed(h, w)
        x = x + pos_embed

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                outputs.append(x)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "drop_path_rate": self.drop_path_rate,
                "q_pool": self.q_pool,
                "q_stride": self.q_stride,
                "stages": self.stages,
                "dim_mul": self.dim_mul,
                "head_mul": self.head_mul,
                "window_pos_embed_bkg_spatial_size": (
                    self.window_pos_embed_bkg_spatial_size
                ),
                "window_spec": self.window_spec,
                "global_att_blocks": self.global_att_blocks,
                "return_interm_layers": self.return_interm_layers,
            }
        )
        return config


class SAM2MemoryAttentionLayer(keras.layers.Layer):
    """A single layer of Memory Attention for SAM2.

    Consists of Self-Attention, Cross-Attention, and a Feedforward Network.

    Args:
        d_model: int. Feature dimension.
        num_heads: int. Number of attention heads.
        dim_feedforward: int. Hidden dimension of MLP.
        dropout: float. Dropout rate.
        activation: str. Activation function.
        pos_enc_at_attn: bool. Whether to add PE in self-attention.
        pos_enc_at_cross_attn_keys: bool. Whether to add PE to keys
        in cross-attention.
        pos_enc_at_cross_attn_queries: bool. Whether to add PE to queries
        in cross-attention.
        rope_theta: float. Theta for RoPE.
        rope_feat_sizes: tuple. Spatial sizes for RoPE cache.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        rope_theta=10000.0,
        rope_feat_sizes=(64, 64),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.rope_theta = rope_theta
        self.rope_feat_sizes = rope_feat_sizes

        self.self_attn = RoPEAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            embedding_dim=d_model,
            rope_theta=rope_theta,
            rope_k_repeat=False,
            feat_sizes=rope_feat_sizes,
            dropout=dropout,
            dtype=self.dtype_policy,
        )
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )
        self.dropout1 = keras.layers.Dropout(dropout, dtype=self.dtype_policy)

        self.cross_attn = RoPEAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            embedding_dim=d_model,
            rope_theta=rope_theta,
            rope_k_repeat=True,
            feat_sizes=rope_feat_sizes,
            dropout=dropout,
            dtype=self.dtype_policy,
        )
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )
        self.dropout2 = keras.layers.Dropout(dropout, dtype=self.dtype_policy)

        self.norm3 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )
        self.linear1 = keras.layers.Dense(
            dim_feedforward, dtype=self.dtype_policy
        )
        self.dropout_mlp = keras.layers.Dropout(
            dropout, dtype=self.dtype_policy
        )
        self.linear2 = keras.layers.Dense(d_model, dtype=self.dtype_policy)
        self.activation_layer = keras.layers.Activation(
            activation, dtype=self.dtype_policy
        )

        self.dropout3 = keras.layers.Dropout(dropout, dtype=self.dtype_policy)

    def build(self, input_shape=None):
        self.self_attn.build()
        self.norm1.build((None, None, self.d_model))
        self.cross_attn.build()
        self.norm2.build((None, None, self.d_model))
        self.norm3.build((None, None, self.d_model))

        self.linear1.build((None, None, self.d_model))
        self.linear2.build((None, None, self.dim_feedforward))

        self.built = True

    def call(self, tgt, memory, pos=None, query_pos=None, num_k_exclude_rope=0):
        tgt2 = self.norm1(tgt)

        q = k = tgt2
        if self.pos_enc_at_attn and query_pos is not None:
            q = q + query_pos
            k = k + query_pos

        tgt2 = self.self_attn(query=q, key=k, value=tgt2)
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)

        q = tgt2
        if self.pos_enc_at_cross_attn_queries and query_pos is not None:
            q = q + query_pos

        k = memory
        if self.pos_enc_at_cross_attn_keys and pos is not None:
            k = k + pos

        tgt2 = self.cross_attn(
            query=q, key=k, value=memory, num_k_exclude_rope=num_k_exclude_rope
        )
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear1(tgt2)
        tgt2 = self.activation_layer(tgt2)
        tgt2 = self.dropout_mlp(tgt2)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "activation": self.activation,
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


class NoMemEmbed(keras.layers.Layer):
    """Layer to add learnable 'no memory' embeddings to features.

    This layer introduces learnable embeddings (`no_mem_embed` and
    `no_mem_pos_enc`) that are added to the input vision features and
    positional encodings. This is typically used in SAM2 for the initial
    frame where no past memory is available.

    Args:
        d_model: int. The feature dimension of the embeddings.

    """

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.no_mem_embed = self.add_weight(
            name="no_mem_embed",
            shape=(1, 1, 1, d_model),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
        )
        self.no_mem_pos_enc = self.add_weight(
            name="no_mem_pos_enc",
            shape=(1, 1, 1, d_model),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
        )

    def call(self, vision_features, vision_pos_enc):
        return (
            vision_features + self.no_mem_embed,
            vision_pos_enc + self.no_mem_pos_enc,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config
