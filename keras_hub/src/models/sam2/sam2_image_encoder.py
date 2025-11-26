import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.sam2.sam2_layers import FpnNeck
from keras_hub.src.models.sam2.sam2_layers import Hiera


@keras_hub_export("keras_hub.models.SAM2ImageEncoder")
class SAM2ImageEncoder(keras.layers.Layer):
    """SAM2 Image Encoder (Backbone).

    Consists of a Hiera trunk and an FPN Neck.

    Args:
        embed_dim: int. Initial embedding dimension for Hiera.
        num_heads: int. Initial number of heads for Hiera.
        drop_path_rate: float. Stochastic depth rate.
        stages: tuple. Number of blocks per stage in Hiera.
        global_att_blocks: tuple. Indices of blocks with global attention.
        window_pos_embed_bkg_spatial_size: tuple. Spatial size for
        background pos embed.
        window_spec: tuple. Window sizes per stage.
        d_model: int. Output dimension for FPN neck.
        backbone_channel_list: list. Channel dimensions for backbone outputs.
        fpn_top_down_levels: list. Levels for FPN top-down path.
        scalp: int. Number of lowest-resolution features to discard.
    """

    def __init__(
        self,
        embed_dim=96,
        num_heads=1,
        drop_path_rate=0.0,
        stages=(2, 3, 16, 3),
        global_att_blocks=(12, 16, 20),
        window_pos_embed_bkg_spatial_size=(14, 14),
        window_spec=(8, 4, 14, 7),
        d_model=256,
        backbone_channel_list=[96, 192, 384, 768],
        fpn_top_down_levels=None,
        scalp=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.trunk = Hiera(
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            stages=stages,
            global_att_blocks=global_att_blocks,
            window_pos_embed_bkg_spatial_size=window_pos_embed_bkg_spatial_size,
            window_spec=window_spec,
            return_interm_layers=True,
            dtype=self.dtype_policy,
        )

        self.neck = FpnNeck(
            d_model=d_model,
            backbone_channel_list=backbone_channel_list,
            fpn_top_down_levels=fpn_top_down_levels,
            dtype=self.dtype_policy,
        )
        self.scalp = scalp

    def build(self, input_shape):
        self.trunk.build(input_shape)
        neck_input_shapes = [
            (None, None, None, c) for c in self.neck.backbone_channel_list
        ]
        self.neck.build(neck_input_shapes)
        self.built = True

    def call(self, x):
        features = self.trunk(x)
        features, pos = self.neck(features)

        if self.scalp > 0:
            features = features[: -self.scalp]
            pos = pos[: -self.scalp]

        return {
            "vision_features": features[-1],
            "vision_pos_enc": pos[-1],
            "backbone_fpn": features,
            "backbone_pos_enc": pos,
        }

    def get_config(self):
        config = super().get_config()
        trunk_config = self.trunk.get_config()
        neck_config = self.neck.get_config()

        config.update(
            {
                "embed_dim": trunk_config["embed_dim"],
                "num_heads": trunk_config["num_heads"],
                "drop_path_rate": trunk_config["drop_path_rate"],
                "stages": trunk_config["stages"],
                "global_att_blocks": trunk_config["global_att_blocks"],
                "window_pos_embed_bkg_spatial_size": trunk_config[
                    "window_pos_embed_bkg_spatial_size"
                ],
                "window_spec": trunk_config["window_spec"],
                "d_model": neck_config["d_model"],
                "backbone_channel_list": neck_config["backbone_channel_list"],
                "fpn_top_down_levels": neck_config["fpn_top_down_levels"],
                "scalp": self.scalp,
            }
        )
        return config
