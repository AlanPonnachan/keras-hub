import numpy as np
import pytest

from keras_hub.src.models.sam2.sam2_backbone import SAM2Backbone
from keras_hub.src.models.sam2.sam2_image_converter import SAM2ImageConverter
from keras_hub.src.models.sam2.sam2_image_encoder import SAM2ImageEncoder
from keras_hub.src.models.sam2.sam2_image_segmenter import SAM2ImageSegmenter
from keras_hub.src.models.sam2.sam2_image_segmenter_preprocessor import (
    SAM2ImageSegmenterPreprocessor,
)
from keras_hub.src.models.sam2.sam2_mask_decoder import SAM2MaskDecoder
from keras_hub.src.models.sam2.sam2_memory_attention import SAM2MemoryAttention
from keras_hub.src.models.sam2.sam2_memory_encoder import SAM2MemoryEncoder
from keras_hub.src.models.sam2.sam2_prompt_encoder import SAMPromptEncoder
from keras_hub.src.tests.test_case import TestCase


class SAM2ImageSegmenterTest(TestCase):
    def setUp(self):
        self.image_size = 64
        self.batch_size = 2

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
            input_image_size=(self.image_size, self.image_size),
            mask_in_channels=4,
        )

        self.mask_decoder = SAM2MaskDecoder(
            hidden_size=16,
            num_layers=1,
            intermediate_dim=32,
            num_heads=2,
            embedding_dim=16,
            num_multimask_outputs=3,
            pred_obj_scores=True,
        )

        self.backbone = SAM2Backbone(
            image_encoder=self.image_encoder,
            memory_attention=self.memory_attention,
            memory_encoder=self.memory_encoder,
            prompt_encoder=self.prompt_encoder,
            mask_decoder=self.mask_decoder,
        )

        self.image_converter = SAM2ImageConverter(
            height=self.image_size, width=self.image_size, scale=1 / 255.0
        )
        self.preprocessor = SAM2ImageSegmenterPreprocessor(self.image_converter)

        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }

        self.inputs = {
            "images": np.ones(
                (self.batch_size, self.image_size, self.image_size, 3),
                dtype="float32",
            ),
            "points": np.ones((self.batch_size, 1, 2), dtype="float32"),
            "labels": np.ones((self.batch_size, 1), dtype="int32"),
            "boxes": np.ones((self.batch_size, 1, 2, 2), dtype="float32"),
            "masks": np.zeros((self.batch_size, 16, 16, 1), dtype="float32"),
        }

        self.train_data = (
            self.inputs,
            {"masks": np.ones((self.batch_size, 4, 16, 16), dtype="float32")},
        )

    def test_sam2_basics(self):
        pytest.skip("TODO: enable after preprocessor flow is figured out")
        self.run_task_test(
            cls=SAM2ImageSegmenter,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape={
                "masks": (self.batch_size, 4, 16, 16),
                "iou_pred": (self.batch_size, 4),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SAM2ImageSegmenter,
            init_kwargs=self.init_kwargs,
            input_data=self.inputs,
        )

    def test_end_to_end_model_predict(self):
        model = SAM2ImageSegmenter(**self.init_kwargs)
        outputs = model.predict(self.inputs)
        masks = outputs["masks"]
        iou_pred = outputs["iou_pred"]

        self.assertEqual(masks.shape, (self.batch_size, 4, 16, 16))
        self.assertEqual(iou_pred.shape, (self.batch_size, 4))

        if "object_score_logits" in outputs:
            self.assertEqual(
                outputs["object_score_logits"].shape, (self.batch_size, 1)
            )

    @pytest.mark.extra_large
    def test_all_presets(self):
        if (
            not hasattr(SAM2ImageSegmenter, "presets")
            or not SAM2ImageSegmenter.presets
        ):
            pytest.skip("No presets available")

        for preset in SAM2ImageSegmenter.presets:
            self.run_preset_test(
                cls=SAM2ImageSegmenter,
                preset=preset,
                input_data=self.inputs,
                expected_output_shape={
                    "masks": (self.batch_size, 4, None, None),
                    "iou_pred": (self.batch_size, 4),
                },
            )
