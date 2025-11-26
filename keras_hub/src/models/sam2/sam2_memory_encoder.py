import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.sam2.sam2_layers import CXBlock
from keras_hub.src.models.sam2.sam2_layers import (
    FixedFrequencyPositionalEmbeddings,
)
from keras_hub.src.models.sam2.sam2_layers import MaskDownSampler


@keras_hub_export("keras_hub.layers.SAM2MemoryEncoder")
class SAM2MemoryEncoder(keras.layers.Layer):
    """Memory Encoder for SAM2.

    Encodes the image features and the predicted mask into a memory feature
    and positional encoding.

    Args:
        out_dim: int. Output dimension of the memory features.
        in_dim: int. Input dimension of the pixel features.
        mask_downsampler_kernel_size: int. Kernel size for mask downsampler.
        mask_downsampler_stride: int. Stride for mask downsampler.
        mask_downsampler_total_stride: int. Total stride for mask downsampler.
        fuser_num_layers: int. Number of layers in the fuser block.
        fuser_dim: int. Dimension of the fuser block.
        fuser_kernel_size: int. Kernel size for CXBlocks in fuser.
        fuser_drop_path: float. Stochastic depth rate.
        activation: str. Activation function.
    """

    def __init__(
        self,
        out_dim,
        in_dim=256,
        mask_downsampler_kernel_size=3,
        mask_downsampler_stride=2,
        mask_downsampler_total_stride=4,
        fuser_num_layers=2,
        fuser_dim=256,
        fuser_kernel_size=7,
        fuser_drop_path=0.0,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mask_downsampler_kernel_size = mask_downsampler_kernel_size
        self.mask_downsampler_stride = mask_downsampler_stride
        self.mask_downsampler_total_stride = mask_downsampler_total_stride
        self.fuser_num_layers = fuser_num_layers
        self.fuser_dim = fuser_dim
        self.fuser_kernel_size = fuser_kernel_size
        self.fuser_drop_path = fuser_drop_path
        self.activation = activation

        self.mask_downsampler = MaskDownSampler(
            embed_dim=in_dim,
            kernel_size=mask_downsampler_kernel_size,
            stride=mask_downsampler_stride,
            padding="same",
            total_stride=mask_downsampler_total_stride,
            activation=activation,
            dtype=self.dtype_policy,
        )

        self.pix_feat_proj = keras.layers.Conv2D(
            filters=in_dim, kernel_size=1, dtype=self.dtype_policy
        )

        self.fuser_layers = []
        for _ in range(fuser_num_layers):
            self.fuser_layers.append(
                CXBlock(
                    dim=fuser_dim,
                    kernel_size=fuser_kernel_size,
                    padding="same",
                    drop_path=fuser_drop_path,
                    use_dwconv=True,
                    dtype=self.dtype_policy,
                )
            )

        if out_dim != in_dim:
            self.out_proj = keras.layers.Conv2D(
                filters=out_dim, kernel_size=1, dtype=self.dtype_policy
            )
        else:
            self.out_proj = None

        self.position_encoding = FixedFrequencyPositionalEmbeddings(
            num_positional_features=out_dim,
            normalize=True,
            dtype=self.dtype_policy,
        )

    def build(self, pix_feat_shape=None, masks_shape=None):
        self.mask_downsampler.build((None, None, None, 1))

        self.pix_feat_proj.build((None, None, None, self.in_dim))

        fuser_shape = (None, None, None, self.fuser_dim)
        for layer in self.fuser_layers:
            layer.build(fuser_shape)

        if self.out_proj:
            self.out_proj.build((None, None, None, self.in_dim))

        self.position_encoding.build(None)
        self.built = True

    def call(self, pix_feat, masks, skip_mask_sigmoid=False):
        if not skip_mask_sigmoid:
            masks = ops.sigmoid(masks)

        masks = self.mask_downsampler(masks)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks

        for layer in self.fuser_layers:
            x = layer(x)

        if self.out_proj:
            x = self.out_proj(x)

        pos = self.position_encoding(x)

        return {"vision_features": x, "vision_pos_enc": pos}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_dim": self.out_dim,
                "in_dim": self.in_dim,
                "mask_downsampler_kernel_size": (
                    self.mask_downsampler_kernel_size
                ),
                "mask_downsampler_stride": self.mask_downsampler_stride,
                "mask_downsampler_total_stride": (
                    self.mask_downsampler_total_stride
                ),
                "fuser_num_layers": self.fuser_num_layers,
                "fuser_dim": self.fuser_dim,
                "fuser_kernel_size": self.fuser_kernel_size,
                "fuser_drop_path": self.fuser_drop_path,
                "activation": self.activation,
            }
        )
        return config
