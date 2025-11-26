import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.sam2.sam2_layers import NoMemEmbed


@keras_hub_export("keras_hub.models.SAM2Backbone")
class SAM2Backbone(Backbone):
    """A backbone for the Segment Anything Model 2 (SAM2).

    This backbone connects the Image Encoder, Prompt Encoder, Memory Attention,
    Memory Encoder, and Mask Decoder. The functional graph defined by this class
    represents the "Single Image" forward pass (or the first frame of a video),
    where no previous memory is present.

    Args:
        image_encoder: `keras_hub.models.SAM2ImageEncoder`. Feature extractor
            for input images.
        memory_attention: `keras_hub.layers.SAM2MemoryAttention`. Layer to
            condition features on memory.
        memory_encoder: `keras_hub.layers.SAM2MemoryEncoder`. Layer to encode
            outputs into memory.
        prompt_encoder: `keras_hub.models.SAMPromptEncoder`. Layer to compute
            embeddings for points, box, and mask prompts.
        mask_decoder: `keras_hub.layers.SAM2MaskDecoder`. Layer to generate
            segmentation masks.
        dtype: The dtype of the layer weights.

    Example:
    ```python
    image_size = 1024
    batch_size = 2
    input_data = {
        "images": np.ones(
            (batch_size, image_size, image_size, 3),
            dtype="float32",
        ),
        "points": np.ones((batch_size, 1, 2), dtype="float32"),
        "labels": np.ones((batch_size, 1), dtype="float32"),
        "boxes": np.ones((batch_size, 1, 2, 2), dtype="float32"),
        "masks": np.zeros(
            (batch_size, 0, image_size, image_size, 1)
        ),
    }

    # 1. Define Sub-Models (e.g., Tiny Config)
    image_encoder = keras_hub.models.SAM2ImageEncoder(
        embed_dim=96, num_heads=1, stages=(1, 2, 7, 2),
        global_att_blocks=(5, 7, 9), window_pos_embed_bkg_spatial_size=(7, 7),
        window_spec=(8, 4, 14, 7), d_model=256,
        backbone_channel_list=[96, 192, 384, 768], scalp=1
    )
    memory_attention = keras_hub.layers.SAM2MemoryAttention(
        d_model=256, num_layers=4, num_heads=1, rope_feat_sizes=(64, 64)
    )
    memory_encoder = keras_hub.layers.SAM2MemoryEncoder(
        out_dim=64, in_dim=256, fuser_num_layers=2
    )
    prompt_encoder = keras_hub.models.SAMPromptEncoder(
        hidden_size=256, image_embedding_size=(64, 64),
        input_image_size=(1024, 1024), mask_in_channels=16
    )
    mask_decoder = keras_hub.layers.SAM2MaskDecoder(
        hidden_size=256, num_layers=2, intermediate_dim=2048, num_heads=8,
        embedding_dim=256, num_multimask_outputs=3, pred_obj_scores=True
    )

    # 2. Create Backbone
    backbone = keras_hub.models.SAM2Backbone(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )

    # 3. Call
    outputs = backbone(input_data)
    ```
    """

    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        prompt_encoder,
        mask_decoder,
        dtype=None,
        **kwargs,
    ):
        # === Sub-models ===
        self.image_encoder = image_encoder
        self.memory_attention = memory_attention
        self.memory_encoder = memory_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        # === Helper Layer for No-Mem Weights ===
        d_model = self.image_encoder.neck.d_model
        self.no_mem_layer = NoMemEmbed(d_model=d_model, dtype=dtype)

        # === Functional Graph (Single Image Pass) ===
        # Inputs
        inputs = {
            "images": keras.Input(shape=(None, None, 3), name="images"),
            "points": keras.Input(shape=(None, 2), name="points"),
            "labels": keras.Input(shape=(None,), dtype="int32", name="labels"),
            "boxes": keras.Input(shape=(None, 2, 2), name="boxes"),
            "masks": keras.Input(shape=(None, None, None, 1), name="masks"),
        }

        img_features = self.image_encoder(inputs["images"])

        prompt_outputs = self.prompt_encoder(
            points=inputs["points"],
            labels=inputs["labels"],
            boxes=inputs["boxes"],
            masks=inputs["masks"],
        )

        vision_features, vision_pos_enc = self.no_mem_layer(
            img_features["vision_features"], img_features["vision_pos_enc"]
        )

        outputs = {
            "vision_features": vision_features,
            "vision_pos_enc": vision_pos_enc,
            "backbone_fpn": img_features["backbone_fpn"],
            "backbone_pos_enc": img_features["backbone_pos_enc"],
        }
        outputs.update(prompt_outputs)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_encoder": keras.layers.serialize(self.image_encoder),
                "memory_attention": keras.layers.serialize(
                    self.memory_attention
                ),
                "memory_encoder": keras.layers.serialize(self.memory_encoder),
                "prompt_encoder": keras.layers.serialize(self.prompt_encoder),
                "mask_decoder": keras.layers.serialize(self.mask_decoder),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "image_encoder": keras.layers.deserialize(
                    config["image_encoder"]
                ),
                "memory_attention": keras.layers.deserialize(
                    config["memory_attention"]
                ),
                "memory_encoder": keras.layers.deserialize(
                    config["memory_encoder"]
                ),
                "prompt_encoder": keras.layers.deserialize(
                    config["prompt_encoder"]
                ),
                "mask_decoder": keras.layers.deserialize(
                    config["mask_decoder"]
                ),
            }
        )
        return super().from_config(config)
