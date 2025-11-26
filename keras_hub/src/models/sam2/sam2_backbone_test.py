import numpy as np

from keras_hub.src.models.sam2.sam2_backbone import SAM2Backbone
from keras_hub.src.models.sam2.sam2_image_encoder import SAM2ImageEncoder
from keras_hub.src.models.sam2.sam2_mask_decoder import SAM2MaskDecoder
from keras_hub.src.models.sam2.sam2_memory_attention import SAM2MemoryAttention
from keras_hub.src.models.sam2.sam2_memory_encoder import SAM2MemoryEncoder
from keras_hub.src.models.sam2.sam2_prompt_encoder import SAMPromptEncoder
from keras_hub.src.tests.test_case import TestCase


class SAM2BackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = 64
        self.embed_dim = 16

        self.image_encoder = SAM2ImageEncoder(
            embed_dim=16,
            num_heads=1,
            stages=(1, 1, 1, 1),
            global_att_blocks=(2,),
            window_pos_embed_bkg_spatial_size=(4, 4),
            window_spec=(8, 4, 4, 2),
            d_model=16,
            backbone_channel_list=[16, 32, 64, 128],
            scalp=1,
        )

        self.memory_attention = SAM2MemoryAttention(
            d_model=16, num_layers=1, num_heads=2, rope_feat_sizes=(4, 4)
        )

        self.memory_encoder = SAM2MemoryEncoder(
            out_dim=16, in_dim=16, fuser_num_layers=1
        )

        self.prompt_encoder = SAMPromptEncoder(
            hidden_size=16,
            image_embedding_size=(4, 4),
            input_image_size=(64, 64),
            mask_in_channels=4,
        )

        self.mask_decoder = SAM2MaskDecoder(
            hidden_size=16,
            num_layers=1,
            intermediate_dim=32,
            num_heads=2,
            embedding_dim=16,
        )

        self.init_kwargs = {
            "image_encoder": self.image_encoder,
            "memory_attention": self.memory_attention,
            "memory_encoder": self.memory_encoder,
            "prompt_encoder": self.prompt_encoder,
            "mask_decoder": self.mask_decoder,
        }

        self.input_data = {
            "images": np.ones(
                (self.batch_size, self.image_size, self.image_size, 3),
                dtype="float32",
            ),
            "points": np.ones((self.batch_size, 2, 2), dtype="float32"),
            "labels": np.ones((self.batch_size, 2), dtype="int32"),
            "boxes": np.ones((self.batch_size, 2, 2, 2), dtype="float32"),
            "masks": np.ones((self.batch_size, 16, 16, 1), dtype="float32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=SAM2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "vision_features": (self.batch_size, 4, 4, 16),
                "vision_pos_enc": (self.batch_size, 4, 4, 16),
                "prompt_sparse_embeddings": (self.batch_size, 6, 16),
                "prompt_dense_embeddings": (self.batch_size, 4, 4, 16),
                "prompt_dense_positional_embeddings": (
                    self.batch_size,
                    4,
                    4,
                    16,
                ),
            },
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    def test_forward_pass(self):
        backbone = SAM2Backbone(**self.init_kwargs)
        outputs = backbone(self.input_data)

        self.assertEqual(len(outputs["backbone_fpn"]), 3)
        self.assertEqual(len(outputs["backbone_pos_enc"]), 3)

        self.assertEqual(
            outputs["vision_features"].shape, (self.batch_size, 4, 4, 16)
        )

        self.assertEqual(
            outputs["prompt_sparse_embeddings"].shape, (self.batch_size, 6, 16)
        )
